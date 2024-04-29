import copy
import math
from typing import Optional, Tuple
from einops import rearrange
import torch 
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaIntermediate, RobertaOutput, apply_chunking_to_forward
from diffus_ie.models.embeddings import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusIEBlock(nn.Module):
    """
    [x]TODO: Implement adaLN-Zero (Scalable Diffusion Models with Transformers)
    """
    def __init__(self, config, block_type: str = 'in-context'):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.self_attention = RobertaSelfAttention(config)
        self.self_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.self_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate = RobertaIntermediate(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(3 * config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        self.block_type = block_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        if self.block_type=='adaLN-Zero':
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition_emb).chunk(6, dim=1)
            _hidden_states = modulate(hidden_states, shift_msa, scale_msa) # scale, shift
            self_outputs = self.self_attention(
                                            _hidden_states,
                                            attention_mask,
                                            head_mask,
                                            self_attn_past_key_value,
                                            output_attentions,
                                        )[0]
            self_outputs = self.self_dense(self_outputs)
            self_outputs = self.self_dropout(self_outputs)
            self_outputs = gate_msa.unsqueeze(1) * self_outputs # scale
            self_outputs = self.self_LayerNorm(self_outputs + hidden_states)

            _self_outputs = modulate(self_outputs, shift_mlp, scale_mlp) # scale and shift
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, _self_outputs
            )
            layer_output = gate_mlp.unsqueeze(1) * layer_output  # scale
            layer_output = self.LayerNorm(self_outputs + layer_output)
        else:
            size = hidden_states.size()
            hidden_states = torch.cat([hidden_states, condition_emb], dim=-2)
            self_outputs = self.self_attention(
                                            hidden_states,
                                            attention_mask,
                                            head_mask,
                                            self_attn_past_key_value,
                                            output_attentions,
                                        )[0]
            self_outputs = self.self_dense(self_outputs)
            self_outputs = self.self_dropout(self_outputs)
            self_outputs = self.self_LayerNorm(self_outputs + hidden_states)
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, self_outputs
            )
            layer_output = self.LayerNorm(self_outputs + layer_output)
            layer_output = layer_output[:, :size[1], :]

        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.dense(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        return intermediate_output


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_size, block_type='in-context'):
        super().__init__()
        self.block_type = block_type
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        if block_type == 'adaLN-Zero':
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(3 * hidden_size, 2 * hidden_size, bias=True)
            )
        else:
            self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

    def forward(self, x, c):
        if self.block_type == 'adaLN-Zero':
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x, _ = self.multihead_attn(query=x, key=c, value=c)
        x = self.linear(x)
        return x


class DiffusIE(nn.Module):
    def __init__(self, params, output_size=None, *args, **kwargs) -> None:
        super().__init__()
        self.params = params
        self.plm_model_name = params.model_name
        self.block_type = params.block_type
        self.depth = params.diff_depth
        if output_size != None:
            self.output_size = output_size
        self.init_model()

    def init_model(self): 
        plm_model = AutoModel.from_pretrained(self.plm_model_name, cache_dir=self.params.hf_cache)
        plm_config = plm_model.config
        if not hasattr(self, 'output_size'):
            self.output_size = plm_config.hidden_size

        self.t_embedder = TimestepEmbedder(plm_config.hidden_size)
        self.blocks = nn.ModuleList([
            DiffusIEBlock(config=plm_model.config, block_type=self.block_type) for _ in range(self.depth)
        ])
        self.final_layer = FinalLayer(plm_model.config.hidden_size, self.output_size, self.block_type)

        # initialize weights
        # timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # diffusion model
        for i, block in enumerate(self.blocks):
            # plm_layer = plm_model.encoder.layer[i]
            # plm_state_dict = plm_layer.state_dict()
            # mapping = {
            #     'attention.self.query.weight': 'self_attention.query.weight',
            #     'attention.self.query.bias': 'self_attention.query.bias',
            #     'attention.self.key.weight': 'self_attention.key.weight',
            #     'attention.self.key.bias': 'self_attention.key.bias',
            #     'attention.self.value.weight': 'self_attention.value.weight',
            #     'attention.self.value.bias': 'self_attention.value.bias',
            #     'attention.output.dense.weight' : 'self_dense.weight', 
            #     'attention.output.dense.bias': 'self_dense.bias', 
            #     'attention.output.LayerNorm.weight': 'self_LayerNorm.weight', 
            #     'attention.output.LayerNorm.bias': 'self_LayerNorm.bias',
            #     'intermediate.dense.weight': 'intermediate.dense.weight', 
            #     'intermediate.dense.bias': 'intermediate.dense.bias',
            #     'output.dense.weight': 'dense.weight', 
            #     'output.dense.bias': 'dense.bias', 
            #     'output.LayerNorm.weight': 'LayerNorm.weight', 
            #     'output.LayerNorm.bias': 'LayerNorm.bias'
            # }
            # state_dict = {mapping[k]: v for k, v in plm_state_dict.items() if k in mapping}
            # block_state_dict = block.state_dict()
            # block_state_dict.update(state_dict)
            # block.load_state_dict(block_state_dict)
        
            # Zero-out adaLN modulation layers in DiT blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.block_type == 'adaLN-Zero':
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        # nn.init.normal_(self.final_layer.linear.bias, std=0.02)

    def forward(self, 
                noisy_emb: torch.Tensor,
                condition_emb: torch.Tensor,
                timestep: torch.Tensor):
        """
        Forward pass of DiffusIE
        noisy_emb: (bs, label_max_len, hidden_dim) tensor of noisy latent label presentation
        condition_emb: (bs, 1, hidden_dim) tensor of condition embedding (normally is the presentation of task-tokens, e.g. event-pair in EERE, trigger in ED, or all sentence in SA)
        timestep: (N,) tensor of diffusion timestep
        """
        time_emb = self.t_embedder(timestep)
        if self.block_type == 'adaLN-Zero':
            condition_emb = torch.cat([time_emb.unsqueeze(-2), condition_emb], dim=-2)
            condition_emb = rearrange(condition_emb, 'b l h -> b (l h)')
        else:
            condition_emb = torch.cat([time_emb.unsqueeze(-2), condition_emb], dim=-2) # (bs, l, hidden_dim)

        for block in self.blocks:
            noisy_emb = block(noisy_emb, condition_emb) 
            
        noise = self.final_layer(noisy_emb, condition_emb)

        return noise

