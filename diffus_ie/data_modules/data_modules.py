import os
from collections import defaultdict
from typing import List, Tuple
import datasets
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer
from diffus_ie.data_modules.constants import CTB_LABEL, ESL_LABEL, MECI_LABEL
from diffus_ie.data_modules.data_preparer import load


class EREDataModule(pl.LightningDataModule):
    def __init__(self, params, 
                 fold: int=None) -> None:
        super().__init__()
        self.params = params
        self.dataname = self.params.data_name 
        if self.dataname == 'ESL':
            self.label_map = ESL_LABEL
            self.fold = fold
        elif self.dataname == 'Causal-TB':
            self.label_map = CTB_LABEL
            self.fold = fold
        elif self.dataname in ['MECI-en', 'MECI-da', 'MECI-es', 'MECI-tr', 'MECI-ur']:
            self.label_map = MECI_LABEL
            self.fold = None
    
        self.model_name = self.params.model_name
        self.batch_size = self.params.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=params.hf_cache)
    
    def transfrom(self, dataset):
        data = []
        for doc in tqdm.tqdm(dataset):
            relations = doc['relations']
            events = doc['events']
            sentences = doc['sentences']
            for pair, rel in relations.items():
                eid1, eid2 = pair.split('-')[0], pair.split('-')[1] 
                try:
                    e1, e2 = events[eid1], events[eid2]
                    sid1, sid2 = str(e1['sent_id']), str(e2['sent_id'])
                except:
                    print(doc['doc_id'])
                    breakpoint()
                s1, s2 = sentences[sid1], sentences[sid2]
                e1_start_char_in_sent, e1_end_char_in_sent = e1['start_char'] - s1['start_char'], e1['end_char'] - s1['start_char']
                e2_start_char_in_sent, e2_end_char_in_sent = e2['start_char'] - s2['start_char'], e2['end_char'] - s2['start_char']
                if s1['content'][e1_start_char_in_sent: e1_end_char_in_sent] != e1['mention']:
                    print(f"{s1[e1_start_char_in_sent: e1_end_char_in_sent]} - {e1['mention']}")
                if s2['content'][e2_start_char_in_sent: e2_end_char_in_sent] != e2['mention']:
                    print(f"{s2[e2_start_char_in_sent: e2_end_char_in_sent]} - {e2['mention']}")

                surround_sid = set(list(range(int(sid1)-2, int(sid1)+2)) + list(range(int(sid2)-2, int(sid2)+2)))
                surround_sentences = [[str(idx), sentences[str(idx)]] for idx in surround_sid 
                                                            if sentences.get(str(idx)) != None]
                augmented_doc = ""
                _start = 0
                for sid, sentence in surround_sentences:
                    if sid == sid1:
                        s1_start = _start
                    if sid == sid2:
                        s2_start = _start
                    augmented_doc = augmented_doc + sentence['content'] + ' '
                    _start += len(sentence['content']) + 1
                e1_start, e1_end = e1_start_char_in_sent + s1_start, e1_end_char_in_sent + s1_start
                e2_start, e2_end = e2_start_char_in_sent + s2_start, e2_end_char_in_sent + s2_start
                if augmented_doc[e1_start: e1_end] != e1['mention']:
                    print(f"{augmented_doc[e1_start: e1_end]} - {e1['mention']}")
                if augmented_doc[e2_start: e2_end] != e2['mention']:
                    print(f"{augmented_doc[e2_start: e2_end]} - {e2['mention']}")

                e1_start += abs(e1_start - e1_end) - len(augmented_doc[e1_start: e1_end].lstrip())
                e2_start += abs(e2_start - e2_end) - len(augmented_doc[e2_start: e2_end].lstrip())
                input_ids, e1_index, e2_index = self.tokenize(augmented_doc, e1_start, e2_start)
                data.append({
                            "context": augmented_doc,
                            "input_ids": input_ids,
                            'e1_mention': e1['mention'],
                            'e2_mention': e2['mention'],
                            "e1_index": e1_index,
                            "e2_index": e2_index,
                            "e1_start": e1_start,
                            "e1_end": e1_end,
                            "e2_start": e2_start,
                            "e2_end": e2_end,
                            "relation": rel
                })
        return data
    
    def tokenize(self, text: str, e1_start_char: int, e2_start_char: int):
        output = self.tokenizer(text=text)
        ids = output.input_ids
        e1_index = output.char_to_token(e1_start_char)
        e2_index = output.char_to_token(e2_start_char)
        assert e1_index != None
        assert e2_index != None
        return ids, e1_index, e2_index

    def prepare_data(self) -> None:
        if hasattr(self, 'fold'):
            self.cache_path = os.path.join(self.params.cache, f"{self.dataname}_intra_{self.params.intra}_inter_{self.params.inter}_fold_{self.fold}")
        else:
            self.cache_path = os.path.join(self.params.cache, f"{self.dataname}_intra_{self.params.intra}_inter_{self.params.inter}")
        try:
            dataset_dict = datasets.load_from_disk(self.cache_path)
        except:
            data = load(dataset=self.dataname, intra=self.params.intra, inter=self.params.inter)
            if self.fold == None:
                train_data, dev_data, test_data = data['train'], data['dev'], data['test']
            else:
                train_data, dev_data = data[str(self.fold)]['train'], data[str(self.fold)]['dev']
                test_data = data.get('test')
                if test_data == None:
                    test_data = dev_data

            train_data = self.transfrom(train_data)
            dev_data = self.transfrom(dev_data)
            test_data = self.transfrom(test_data)
            dataset_dict = datasets.DatasetDict({
                                                'train': datasets.Dataset.from_list(train_data),
                                                'dev': datasets.Dataset.from_list(dev_data),
                                                'test': datasets.Dataset.from_list(test_data)})
            dataset_dict.save_to_disk(self.cache_path)
    
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        dataset = datasets.load_from_disk(self.cache_path)
        if stage == "fit":
            self.train_data, self.dev_data = dataset['train'], dataset['dev']

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = dataset['test']

        if stage == "predict":
            raise "Haven't supported!"
    
    def ECI_collate(self, batch):
        """
        input_ids, attn_mask: (bs, max_len) 
        label_ids, label_attn_mask: (bs, max_label_sentence_len)
        trigger_positions: List[Tuple[int, int]] 
        label: (bs,)
        """
        trigger_positions: List[Tuple[int, int]] = [] 
        label = []
        label_sentence = []
        input_ids = []
        
        for item in batch:
            input_ids.append(item['input_ids'])
            trigger_positions.append((item['e1_index'], item['e2_index']))
            _label, template = self.label_map[item['relation']]
            _label_sentence = template.format(e1=item['e1_mention'], e2=item['e2_mention'])
            label.append(_label)
            label_sentence.append(_label_sentence)

        label_tokenized = self.tokenizer(label_sentence, padding='longest', max_length=self.params.label_max_len, return_tensors='pt')
        label_ids, label_attn_mask = label_tokenized.input_ids, label_tokenized.attention_mask
        label = torch.tensor(label)

        input_len = [len(item) for item in input_ids]
        input_ids = pad_sequence([torch.tensor(item) for item in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = torch.zeros_like(input_ids, dtype=torch.int)
        for i, l in enumerate(input_len):
            attn_mask[i, :l] = attn_mask[i, :l] + 1
        
        return input_ids, attn_mask, label_ids, label_attn_mask, trigger_positions, label
    
    def get_collate_fn(self):
        if self.dataname in ['ESL', 'Causal-TB', 'MECI-en', 'MECI-da', 'MECI-es', 'MECI-tr', 'MECI-ur']:
            return self.ECI_collate
        else:
            raise f"We haven't support {self.dataname}!"
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        collate_fn = self.get_collate_fn()
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        collate_fn = self.get_collate_fn()
        return DataLoader(
            dataset=self.dev_data,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        collate_fn = self.get_collate_fn()
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=False,
            shuffle=True,
            collate_fn=collate_fn
        )

