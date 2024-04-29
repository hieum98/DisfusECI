from collections import defaultdict
import json
import os
import pathlib
import random
import re
from sklearn.model_selection import KFold
import tqdm
import datasets
from datasets import DatasetDict, Dataset

from diffus_ie.data_modules.data_reader import cat_xml_reader, ctb_cat_reader, meci_tsvx_reader


class Preprocessor(object):
    def __init__(self, dataset, intra=True, inter=False):
        self.dataset = dataset
        self.intra = intra
        self.inter = inter
        self.register_reader(self.dataset)
        

    def register_reader(self, dataset):
        if dataset == 'ESL':
            self.reader = cat_xml_reader
        elif dataset == 'Causal-TB':
            self.reader = ctb_cat_reader
        elif dataset in ['MECI-en', 'MECI-da', 'MECI-es', 'MECI-tr', 'MECI-ur']:
            self.reader = meci_tsvx_reader
        else:
            raise ValueError("We have not supported this dataset {} yet!".format(self.dataset))
    
    def load_dataset(self, dir_name):
        corpus = []
        if self.dataset == 'ESL':
            topic_folders = [t for t in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, t))]
            for topic in tqdm.tqdm(topic_folders):
                topic_folder = os.path.join(dir_name, topic)
                onlyfiles = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
                for file_name in onlyfiles:
                    file_name = os.path.join(topic, file_name)
                    if file_name.endswith('.xml'):
                        cache_file = pathlib.Path(os.path.join(dir_name, f'cache/{self.inter}_{self.intra}', f'{file_name}.json'))
                        cache_dir = cache_file.parent
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                my_dict = json.load(f)
                        else:
                            my_dict = self.reader(dir_name, file_name, inter=self.inter, intra=self.intra)
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(my_dict, f, indent=4)
                        if my_dict != None:
                            events = {str(e_id): {'mention': value['mention'],
                                             'start_char': value['start_char'],
                                             'end_char': value['end_char'],
                                             'sent_id': value['sent_id']} 
                                        for e_id, value in my_dict['event_dict'].items()}
                            sentences = {str(s['sent_id']): {'content': s['content'],
                                                        'start_char': s['sent_start_char'],
                                                        'end_char': s['sent_end_char'],
                                                        'tokens': s['tokens'],
                                                        'heads': s['heads'],
                                                        'deps': s['deps'],
                                                        'pos': s['pos'],
                                                        'token_span': s['token_span_DOC']} 
                                        for s in my_dict['sentences']}
                            relations = {}
                            for key, rel in my_dict['relation_dict'].items():
                                eid1, eid2 = re.sub('\W+','', key.split(',')[0].strip()), re.sub('\W+','', key.split(',')[1].strip())
                                relations[str(f'{eid1}-{eid2}')] = rel
                            data = {
                                'doc_id': my_dict['doc_id'],
                                'doc_content': my_dict['doc_content'],
                                'events': events,
                                'sentences': sentences,
                                'relations': relations
                            }
                            corpus.append(data)
        else:
            onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            i = 0
            for file_name in tqdm.tqdm(onlyfiles):
                cache_file = pathlib.Path(os.path.join(dir_name, 'cache', f'{file_name}.json'))
                cache_dir = cache_file.parent
                cache_dir.mkdir(parents=True, exist_ok=True)
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8')as f:
                        my_dict = json.load(f)
                else:
                    my_dict = self.reader(dir_name, file_name)
                    with open(cache_file, 'w', encoding='utf-8')as f:
                        json.dump(my_dict, f, indent=4)
                if my_dict != None:
                    events = {str(e_id): {'mention': value['mention'],
                                        'start_char': value['start_char'],
                                        'end_char': value['end_char'],
                                        'sent_id': value['sent_id']} 
                                for e_id, value in my_dict['event_dict'].items()}
                    sentences = {str(s['sent_id']): {'content': s['content'],
                                                'start_char': s['sent_start_char'],
                                                'end_char': s['sent_end_char'],
                                                'tokens': s['tokens'],
                                                'heads': s['heads'],
                                                'deps': s['deps'],
                                                'pos': s['pos'],
                                                'token_span': s['token_span_DOC']} 
                                for s in my_dict['sentences']}
                    relations = {}
                    for key, rel in my_dict['relation_dict'].items():
                        eid1, eid2 = re.sub('\W+','', key.split(',')[0].strip()), re.sub('\W+','', key.split(',')[1].strip())
                        relations[str(f'{eid1}-{eid2}')] = rel
                    data = {
                        'doc_id': my_dict['doc_id'],
                        'doc_content': my_dict['doc_content'],
                        'events': events,
                        'sentences': sentences,
                        'relations': relations
                    }
                    corpus.append(data)

        return corpus
    

def load(dataset: str, load_fold: int=0, intra=True, inter=True):
    if dataset == 'ESL':
        # TODO: Fix the data split to public (val: 2 last topic, train-test: 5 folds)
        kfold = KFold(n_splits=5)
        print(f"Loading ESL intra {intra}, inter {inter}")
        processor = Preprocessor(dataset, intra=intra, inter=inter)
        corpus_dir = '/home/daclai/DiffusECI/data/EventStoryLine/annotated_data/v1.5/'
        corpus = processor.load_dataset(corpus_dir)

        _train, test = [], []
        data = defaultdict(list)
        for my_dict in corpus:
            topic = my_dict['doc_id'].split('/')[0]
            data[topic].append(my_dict)

            if '37/' in my_dict['doc_id'] or '41/' in my_dict['doc_id']:
                test.append(my_dict)
            else:
                _train.append(my_dict)

        random.shuffle(_train)
        folds = {}
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(_train)):
            train = [_train[id] for id in train_ids]
            validate = [_train[id] for id in valid_ids]
            folds[str(fold)] = {'train': train,
                           'dev': validate}
        folds['test'] = test
                
        return folds
    
    if dataset == 'Causal-TB':
        kfold = KFold(n_splits=10)
        print('Loading Causal-TB')
        processor = Preprocessor(dataset)
        corpus_dir = '/home/daclai/DiffusECI/data/Causal-TB/Causal-TB-CAT/'
        corpus = processor.load_dataset(corpus_dir)

        random.shuffle(corpus)
        folds = {}
        for fold, (train_ids, valid_ids) in enumerate(kfold.split(corpus)):
            train = [corpus[id] for id in train_ids]
            validate = [corpus[id] for id in valid_ids]
            folds[str(fold)] = {'train': train,
                           'dev': validate}
        return folds
    
    if dataset in ['MECI-en', 'MECI-da', 'MECI-es', 'MECI-tr', 'MECI-ur']:
        processor = Preprocessor(dataset)
        if dataset == 'MECI-da':
            corpus_dir = '/home/daclai/DiffusECI/data/meci-dataset/meci-v0.1-public/causal-da'
        elif dataset == 'MECI-en':
            corpus_dir = '/home/daclai/DiffusECI/data/meci-dataset/meci-v0.1-public/causal-en'
        elif dataset == 'MECI-es':
            corpus_dir = '/home/daclai/DiffusECI/data/meci-dataset/meci-v0.1-public/causal-es'
        elif dataset == 'MECI-tr':
            corpus_dir = '/home/daclai/DiffusECI/data/meci-dataset/meci-v0.1-public/causal-tr'
        elif dataset == 'MECI-ur':
            corpus_dir = '/home/daclai/DiffusECI/data/meci-dataset/meci-v0.1-public/causal-ur'
        
        train_dir = corpus_dir + '/train/'
        val_dir = corpus_dir + '/dev/'
        test_dir = corpus_dir + '/test/'

        data = {
            'train': processor.load_dataset(train_dir),
            'dev': processor.load_dataset(val_dir),
            'test': processor.load_dataset(test_dir)
        }
        return data
    
    print(f"We have not supported {dataset} dataset!")