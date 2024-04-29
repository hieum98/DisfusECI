from collections import OrderedDict
import copy
from itertools import chain
from typing import Any, List, Optional, Tuple
from einops import rearrange
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from diffusers import DDIMScheduler
import tqdm
from diffus_ie.models.model import DiffusIE


class DiffusIEModel(pl.LightningModule):
    """
    TODO: Implement Improved Denoising Diffusion Probalistic Model (follow DiT paper)
    """
    def __init__(self, params, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.use_diff = params.use_diffusion
        self.num_train_step = params.diffusion_train_step
        self.num_inference_step = params.diffusion_inference_step
        self.batch_size = params.batch_size

        self.encoder_warm_up = params.encoder_warm_up
        self.use_diff = True
        
        if 'base' in params.model_name:
            self.hidden_size = 768
        elif 'large' in params.model_name:
            self.hidden_size = 1024
        self.num_labels = self.params.num_labels
        self.label_max_len = params.label_max_len

        self.encoder = AutoModel.from_pretrained(self.params.model_name, cache_dir=self.params.hf_cache)
        self.model = DiffusIE(params=self.params, output_size=self.hidden_size)
        self.noise_scheduler = DDIMScheduler(self.num_train_step)
        self.noise_scheduler.set_timesteps(self.num_inference_step)
        # attn
        self.proj = nn.Sequential(OrderedDict([('dropout', nn.Dropout(0.3)),
                                            ('mlp', nn.Linear(in_features=self.hidden_size*3, out_features=self.hidden_size*2)),
                                            ('activ', nn.ReLU()),]))
        self.predictor = nn.Sequential(OrderedDict([('dropout_1', nn.Dropout(0.3)),
                                                    ('mlp_1', nn.Linear(in_features=self.hidden_size*2, out_features=self.hidden_size)),
                                                    ('activ_1', nn.ReLU()),
                                                    ('dropout_2', nn.Dropout(0.3)),
                                                    ('mlp_2', nn.Linear(in_features=self.hidden_size, out_features=self.num_labels))]))

        self.task_loss = nn.CrossEntropyLoss()
        self.diffus_loss = nn.MSELoss()

        self.val_outputs = []
        self.val_labels = []
        self.test_outputs = []
        self.test_labels = []

    def configure_optimizers(self) -> Any:
        num_batches = self.trainer.estimated_stepping_batches

        no_decay = ['bias', 'gamma', 'beta', "LayerNorm.weight"]

        parameters = [
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'lr': self.params.diff_lr, 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'lr': self.params.diff_lr, 'weight_decay': 0.00},
            {'params': [p for n, p in self.encoder.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'lr': self.params.encoder_lr, 'weight_decay': 0.01},
            {'params': [p for n, p in self.encoder.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'lr': self.params.encoder_lr, 'weight_decay': 0.00},
            {'params': [p for n, p in self.proj.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'lr': self.params.head_lr, 'weight_decay': 0.01},
            {'params': [p for n, p in self.proj.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'lr': self.params.head_lr, 'weight_decay': 0.00},
            {'params': [p for n, p in self.predictor.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'lr': self.params.head_lr, 'weight_decay': 0.01},
            {'params': [p for n, p in self.predictor.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'lr': self.params.head_lr, 'weight_decay': 0.00},
        ]

        optimizer = torch.optim.AdamW(parameters)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, 
                                                    num_warmup_steps=100,
                                                    num_training_steps=num_batches)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
            },
        }
    
    def compute_pair_emb(self, input_ids, attn_mask, trigger_positions):
        bs = input_ids.size(0)
        output = self.encoder(input_ids=input_ids,
                            attention_mask=attn_mask,
                            return_dict=True,
                            output_hidden_states=True)
        input_emb = output.last_hidden_state # (bs, max_len, hidden_dim)
        pair_emb = []
        for i in range(bs):
            head_position, tail_positon = trigger_positions[i]
            head_emb = input_emb[i, head_position, :]
            tail_emb = input_emb[i, tail_positon, :]
            pair_emb.append(torch.stack([head_emb, tail_emb], dim=0))
        pair_emb = torch.stack(pair_emb, dim=0) # (bs, l, hidden_dim)
        return pair_emb
    
    def sample_label_emb(self, condition_emb):
        """
        [x]TODO: Try different way of aggregate. Max, first, mean, attention. Note: don't need because using attention
        """
        batch_size = condition_emb.size(0)
        latents = torch.rand((batch_size, self.params.label_max_len, self.hidden_size), device=condition_emb.device)
        for i, t in enumerate(self.noise_scheduler.timesteps):
            noise_pred = self.model(latents, condition_emb, torch.zeros(batch_size, device=condition_emb.device) + t)
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        return latents
    
    def metric(self, predicts, labels, verbose=True):
        CM = confusion_matrix(labels, predicts)
        print(f"Confusion Matrix: \n{CM}")
        if verbose:
            print(classification_report(labels, predicts))
        if self.params.data_name in ['ESL', 'Causal-TB', 'MECI-en', 'MECI-da', 'MECI-es', 'MECI-tr', 'MECI-ur']:
            tp = sum([CM[i, i] for i in range(1)]) + 1
            n_preds = sum([CM[i, :-1].sum() for i in range(2)]) + 1 
            t_golds = sum([CM[i].sum() for i in range(1)]) + 1
        else:
            raise "We haven't this dataset yet!"
        
        P = tp/n_preds
        R = tp/t_golds
        F1 = 2 * P * R / (P + R)
        return P, R, F1
    
    def on_train_epoch_start(self) -> None:
        if self.trainer.current_epoch < self.encoder_warm_up:
            self.use_diff = False
        else:
            self.use_diff = True
    
    def training_step(self, batch: Any, batch_idx: Any) -> STEP_OUTPUT:
        input_ids: torch.Tensor = batch[0]        # (bs, max_len)
        attn_mask: torch.Tensor = batch[1]        # (bs, max_len)
        label_ids: torch.Tensor = batch[2]        # (bs, label_max_len)
        label_attn_mask: torch.Tensor = batch[3]
        trigger_positions: List[Tuple[int, int]] = batch[4]   
        label: torch.Tensor = batch[5]            # (bs,)
        bs = input_ids.size(0)
        
        pair_emb = self.compute_pair_emb(input_ids=input_ids,
                                         attn_mask=attn_mask,
                                         trigger_positions=trigger_positions)

        if self.use_diff:
            _label_emb = self.encoder(input_ids=label_ids,
                                    attention_mask=label_attn_mask,
                                    return_dict=True,
                                    output_hidden_states=True).last_hidden_state #(bs, label_max_len, hidden_dim)
            
            label_emb = _label_emb.detach()
            # TODO: Try to project it into a lower space as Stable Diffusion do
            noise = torch.rand(label_emb.shape).to(label_emb.device) # (bs, label_max_length, hidden_dim)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=label_emb.device).long() 
            noisy_label_emb = self.noise_scheduler.add_noise(original_samples=label_emb,
                                                            noise=noise,
                                                            timesteps=timesteps) # (bs, label_max_length, hidden_dim)
            _pair_emb = pair_emb.detach() # prevent backprob to encoder
            noise_pred = self.model(noisy_emb=noisy_label_emb,
                                    condition_emb=_pair_emb,
                                    timestep=timesteps)
            
            diffusion_loss = self.diffus_loss(noise_pred, noise)
            
            latents = []
            for i in range(bs):
                latent = self.noise_scheduler.step(noise_pred[i], timesteps[i], noisy_label_emb[i]).prev_sample
                latents.append(latent)
                
            latents = torch.stack(latents, dim=0) # (bs, label_max_len, hidden_dim)
            #[x]TODO: concatenate with origin pair_emb
            latents = latents[:,0,:]
            pair_emb = rearrange(pair_emb, 'b l h -> b (l h)')
            augemented_pair_emb1 = torch.cat([pair_emb, latents], dim=-1)
            presentation1 = self.proj(augemented_pair_emb1)
            logit1 = self.predictor(presentation1)

            augemented_pair_emb2 = torch.cat([pair_emb, label_emb[:, 0, :]], dim=-1)
            presentation2 = self.proj(augemented_pair_emb2)
            logit2 = self.predictor(presentation2)

            task_loss = 0.9 * self.task_loss(logit1, label) + 0.1 * self.task_loss(logit2, label)
            loss = diffusion_loss + task_loss
            self.log_dict({'loss': loss,
                        'diff_loss': diffusion_loss,
                        'task_loss': task_loss}, prog_bar=True, sync_dist=True)
           
        else:
            presentation = rearrange(pair_emb, 'b l h -> b (l h)')
            logit = self.predictor(presentation)
            task_loss = self.task_loss(logit, label)

            loss = task_loss
            self.log_dict({'loss': loss,
                        'task_loss': task_loss}, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: Any) -> STEP_OUTPUT | None:
        input_ids: torch.Tensor = batch[0]        # (bs, max_len)
        attn_mask: torch.Tensor = batch[1]        # (bs, max_len)
        label_ids: torch.Tensor = batch[2]        # (bs, label_max_len)
        label_attn_mask: torch.Tensor = batch[3]
        trigger_positions: List[Tuple[int, int]] = batch[4]   
        label: torch.Tensor = batch[5]            # (bs,)
        bs = input_ids.size(0)
         
        pair_emb = self.compute_pair_emb(input_ids=input_ids,
                                         attn_mask=attn_mask,
                                         trigger_positions=trigger_positions) # (bs, l, hidden_dim)
        
        if self.use_diff:
            latents = self.sample_label_emb(condition_emb=pair_emb)
            logit = torch.zeros((bs, self.num_labels), device=input_ids.device)
            pair_emb = rearrange(pair_emb, 'b l h -> b (l h)')
            latents = latents[:,0,:]
            augemented_pair_emb = torch.cat([pair_emb, latents], dim=-1)
            presentation = self.proj(augemented_pair_emb)
            logit = F.softmax(self.predictor(presentation), dim=1)
        else:
            presentation = rearrange(pair_emb, 'b l h -> b (l h)')
            logit = F.softmax(self.predictor(presentation), dim=1)

        predicted_label = torch.max(logit, dim=1).indices
        val_task_loss = self.task_loss(logit, label)
        self.val_outputs.append(predicted_label.tolist())
        self.val_labels.append(label.tolist())
        self.log('hp_metric/val_loss', val_task_loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        if len(self.val_labels) > 5:
            predicts = []
            labels = []
            for i in range(len(self.val_labels)):
                predicts.extend(self.val_outputs[i])
                labels.extend(self.val_labels[i])
            P, R, F1 = self.metric(predicts, labels, verbose=True)
            self.log('hp_metric/f1_val', F1, prog_bar=True, sync_dist=True)
            self.log('hp_metric', F1, sync_dist=True)
            
            if not hasattr(self, 'val_result'):
                self.val_result = (P, R, F1)
            if self.val_result[-1] < F1:
                print(f"Best result is updated to {(P, R, F1)}")
                self.val_result = (P, R, F1)
            if F1 < 0.1:
                self.trainer.should_stop = True
        self.val_outputs.clear()
        self.val_labels.clear()
        
    def test_step(self, batch: Any, batch_idx: Any) -> STEP_OUTPUT | None:
        input_ids: torch.Tensor = batch[0]        # (bs, max_len)
        attn_mask: torch.Tensor = batch[1]        # (bs, max_len)
        label_ids: torch.Tensor = batch[2]        # (bs, label_max_len)
        label_attn_mask: torch.Tensor = batch[3]
        trigger_positions: List[Tuple[int, int]] = batch[4]   
        label: torch.Tensor = batch[5]            # (bs,)
        bs = input_ids.size(0)
         
        pair_emb = self.compute_pair_emb(input_ids=input_ids,
                                         attn_mask=attn_mask,
                                         trigger_positions=trigger_positions) # (bs, l, hidden_dim)
        
        if self.use_diff:
            latents = self.sample_label_emb(condition_emb=pair_emb)
            logit = torch.zeros((bs, self.num_labels), device=input_ids.device)
            pair_emb = rearrange(pair_emb, 'b l h -> b (l h)')
            latents = latents[:,0,:]
            augemented_pair_emb = torch.cat([pair_emb, latents], dim=-1)
            presentation = self.proj(augemented_pair_emb)
            logit = F.softmax(self.predictor(presentation), dim=1)
        else:
            presentation = rearrange(pair_emb, 'b l h -> b (l h)')
            logit = F.softmax(self.predictor(presentation), dim=1)

        predicted_label = torch.max(logit, dim=1).indices
        task_loss = self.task_loss(logit, label)
        self.test_outputs.append(predicted_label.tolist())
        self.test_labels.append(label.tolist())
        self.log('test_loss', task_loss, prog_bar=True, sync_dist=True)
    
    def on_test_epoch_end(self) -> None:
        predicts = []
        labels = []
        for i in range(len(self.test_labels)):
            predicts.extend(self.test_outputs[i])
            labels.extend(self.test_labels[i])
        P, R, F1 = self.metric(predicts, labels)
        self.result = (P, R, F1)
        self.log('f1_test', F1, prog_bar=True, sync_dist=True)
        self.test_outputs.clear()
        self.test_labels.clear()
    


