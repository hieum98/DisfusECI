from itertools import combinations
import json
import pdb
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
from bs4 import BeautifulSoup as Soup
import csv
from trankit import Pipeline
from diffus_ie.utils.reader_utils import find_m_id, find_sent_id, get_mention_span, id_lookup, sent_id_lookup, span_SENT_to_DOC, tokenized_to_origin_span

p = Pipeline('english', cache_dir='/home/daclai/DiffusECI/cache')
p.add('danish')
p.add('spanish')
p.add('turkish')
p.add('urdu')


# =========================
#       ESC Reader
# =========================
def cat_xml_reader(dir_name, file_name, intra=True, inter=False):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')

    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None
    
    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['t_id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['content'] = ' '.join(v)
        sent_dict["sent_start_char"] = start
        sent_dict["sent_end_char"] = end = start + len(sent_dict['content'])
        assert sent_dict['content'] == my_dict['doc_content'][start: end]
        
        sent_dict['tokens'] = v
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        sent_dict['pos'] = []
        parsed_tokens = p.posdep(sent_dict['tokens'], is_sent=True)['tokens']
        for token in parsed_tokens:
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(f"This doc {my_dict['doc_id']} was not annotated!")
        return None
    
    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'action' in item.name:
            eid = int(item.attrs['m_id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['t_id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['start_char'], my_dict['event_dict'][eid]['end_char'] = start, start + len(mention)
                my_dict['event_dict'][eid]['token_id'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = sent_id = find_sent_id(my_dict['sentences'], [start, start + len(mention)])
                if not all(tok in  my_dict["event_dict"][eid]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]):
                    print(f'{my_dict["event_dict"][eid]["mention"]} - \
                    {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]}')
                    print(f'{my_dict["event_dict"][eid]}  - {my_dict["sentences"][sent_id]}')
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    event_pairs = list(combinations(my_dict['event_dict'].keys(), 2))
    if intra==True:
        for item in xml_dom.find('relations').children:
            if type(item)==bs4.element.Tag and 'plot_link' in item.name:
                r_id = item.attrs['r_id']
                if item.has_attr('signal'):
                    signal = item.attrs['signal']
                else:
                    signal = ''
                try:
                    r_typ = item.attrs['reltype']
                except:
                    continue
                head = int(item.find('source').attrs['m_id'])
                tail = int(item.find('target').attrs['m_id'])
                if head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys() and len(r_typ.strip()) != 0: # only take event-event relations
                    my_dict['relation_dict'][f"{(head, tail)}"] = r_typ

        # Add Norel into data
        for eid1, eid2 in event_pairs:
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            sid1, sid2 = e1['sent_id'], e2['sent_id']
            if sid1 == sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
                my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'
                
    event_pairs = list(combinations(my_dict['event_dict'].keys(), 2))
    if inter==True:
        dir_name = '/home/daclai/DiffusECI/data/EventStoryLine/annotated_data/v1.5/'
        inter_dir_name = dir_name.replace('annotated_data', 'evaluation_format/full_corpus') + 'event_mentions_extended/'
        file_name = file_name.replace('.xml.xml', '.tab')
        lines = []
        try:
            with open(inter_dir_name+file_name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            print("{} is not exit!".format(inter_dir_name+file_name))
        for line in lines:
            rel = line.strip().split('\t')
            r_typ = rel[2]
            head_span, tail_span = get_mention_span(rel[0]), get_mention_span(rel[1])
            # print(head_span, tail_span)
            head, tail = find_m_id(head_span, my_dict['event_dict']), find_m_id(tail_span, my_dict['event_dict'])
            if head != None and tail != None: 
                if r_typ != 'null':
                    my_dict['relation_dict'][f"{(head, tail)}"] = r_typ
            else: 
                print(f"doc: {inter_dir_name+file_name}, line: {line}, rel: {rel}")

        # Add Norel into data
        for eid1, eid2 in event_pairs:
            e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
            sid1, sid2 = e1['sent_id'], e2['sent_id']
            if sid1 != sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
                my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'
    
    return my_dict


# =========================
#     Causal-TB Reader
# =========================
def ctb_cat_reader(dir_name, file_name):
    my_dict = {}
    my_dict['event_dict'] = {}
    my_dict['doc_id'] = file_name.replace('.xml', '')
    
    try:
        # xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'xml')
        with open(dir_name + file_name, 'r', encoding='UTF-8') as f:
            doc = f.read()
            xml_dom = Soup(doc, 'lxml')
    except Exception as e:
        print("Can't load this file: {}. Please check it T_T". format(dir_name + file_name))
        print(e)
        return None
    
    doc_toks = []
    my_dict['doc_tokens'] = {}
    _sent_dict = defaultdict(list)
    _sent_token_span_doc = defaultdict(list)
    for tok in xml_dom.find_all('token'):
        token = tok.text
        t_id = int(tok.attrs['id'])
        sent_id = int(tok.attrs['sentence'])
        tok_sent_id = len(_sent_dict[sent_id])

        my_dict['doc_tokens'][t_id] = {
            'token': token,
            'sent_id': sent_id,
            'tok_sent_id': tok_sent_id
        }
        
        doc_toks.append(token)
        _sent_dict[sent_id].append(token)
        _sent_token_span_doc[sent_id].append(t_id)
        assert len(doc_toks) == t_id, f"{len(doc_toks)} - {t_id}"
        assert len(_sent_dict[sent_id]) == tok_sent_id + 1
    
    my_dict['doc_content'] = ' '.join(doc_toks)

    my_dict['sentences'] = []
    for k, v in _sent_dict.items():
        start_token_id = _sent_token_span_doc[k][0]
        start = len(' '.join(doc_toks[0:start_token_id-1]))
        if start != 0:
            start = start + 1 # space at the end of the previous sent
        sent_dict = {}
        sent_dict['sent_id'] = k
        sent_dict['content'] = ' '.join(v)
        sent_dict["sent_start_char"] = start
        sent_dict["sent_end_char"] = end = start + len(sent_dict['content'])
        assert sent_dict['content'] == my_dict['doc_content'][start: end]

        sent_dict['tokens'] = v
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []
        sent_dict['pos'] = []
        parsed_tokens = p.posdep(sent_dict['tokens'], is_sent=True)['tokens']
        for token in parsed_tokens:
            sent_dict['pos'].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = parsed_tokens[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent_dict["content"], sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        my_dict["sentences"].append(sent_dict)

    if xml_dom.find('markables') == None:
        print(f"This doc {my_dict['doc_id']} was not annotated!")
        return None
    
    for item in xml_dom.find('markables').children:
        if type(item)== bs4.element.Tag and 'event' in item.name:
            eid = int(item.attrs['id'])
            e_typ = item.name
            mention_span = [int(anchor.attrs['id']) for anchor in item.find_all('token_anchor')]
            mention_span_sent = [my_dict['doc_tokens'][t_id]['tok_sent_id'] for t_id in mention_span]
            
            if len(mention_span) != 0:
                mention = ' '.join(doc_toks[mention_span[0]-1:mention_span[-1]])
                start = len(' '.join(doc_toks[0:mention_span[0]-1]))
                if start != 0:
                    start = start + 1 # space at the end of the previous
                my_dict['event_dict'][eid] = {}
                my_dict['event_dict'][eid]['mention'] = mention
                my_dict['event_dict'][eid]['mention_span'] = mention_span
                my_dict['event_dict'][eid]['start_char'], my_dict['event_dict'][eid]['end_char'] = start, start + len(mention)
                my_dict['event_dict'][eid]['token_id'] = mention_span_sent
                my_dict['event_dict'][eid]['class'] = e_typ
                my_dict['event_dict'][eid]['sent_id'] = sent_id = find_sent_id(my_dict['sentences'], [start, start + len(mention)])
                if not all(tok in  my_dict["event_dict"][eid]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]):
                    print(f'{my_dict["event_dict"][eid]["mention"]} - \
                    {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][eid]["token_id"][0]: my_dict["event_dict"][eid]["token_id"][-1] + 1]}')
                    print(f'{my_dict["event_dict"][eid]}  - {my_dict["sentences"][sent_id]}')
                assert my_dict['event_dict'][eid]['sent_id'] != None
                assert my_dict['doc_content'][start:  start + len(mention)] == mention, f"\n'{mention}' \n'{my_dict['doc_content'][start:  start + len(mention)]}'"
    
    my_dict['relation_dict'] = {}
    for item in xml_dom.find('relations').children:
        if type(item)== bs4.element.Tag and 'clink' in item.name:
            r_id = item.attrs['id']
            r_typ = 'PRECONDITION'
            head = int(item.find('source').attrs['id'])
            tail = int(item.find('target').attrs['id'])

            assert head in my_dict['event_dict'].keys() and tail in my_dict['event_dict'].keys()
            my_dict['relation_dict'][f"{(head, tail)}"] = r_typ
    
    # Add Norel into data
    event_pairs = combinations(my_dict['event_dict'].keys(), 2)
    for eid1, eid2 in event_pairs:
        e1, e2 = my_dict['event_dict'][eid1], my_dict['event_dict'][eid2]
        sid1, sid2 = e1['sent_id'], e2['sent_id']
        if sid1 == sid2 and my_dict['relation_dict'].get(f'{(eid1, eid2)}') == None and my_dict['relation_dict'].get(f'{(eid2, eid1)}') ==None:
            my_dict['relation_dict'][f'{(eid1, eid2)}'] = 'NoRel'

    return my_dict


# =========================
#     MECI Reader
# =========================
def meci_tsvx_reader(dir_name, file_name):
    if 'causal-da' in dir_name:
        p.set_active('danish')
    elif 'causal-en' in dir_name:
        p.set_active('english')
    elif 'causal-es' in dir_name:
        p.set_active('spanish')
    elif 'causal-tr' in dir_name:
        p.set_active('turkish')
    elif 'causal-ur' in dir_name:
        p.set_active('urdu')
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".ann.tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}

    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = '\t'.join(line[1:])
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2])
            my_dict["event_dict"][line[1]] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = line[1]
            event_id2 = line[2]
            rel = line[3]
            my_dict["relation_dict"][f"{(event_id1, event_id2)}"] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # Split document into sentences
    sent_tokenized_text = [sent['text'] for sent in p.ssplit(my_dict["doc_content"])['sentences']]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1

        # Tokenized, Part-Of-Speech Tagging, Dependency Parsing
        _spacy_token = p.posdep(sent_dict["content"], is_sent=True)['tokens']
        spacy_token = []
        for token in _spacy_token:
            if token.get('expanded') == None:
                spacy_token.append(token)
            else:
                spacy_token = spacy_token + token['expanded']

        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict['heads'] = []
        sent_dict['deps'] = []
        sent_dict['idx_char_heads'] = []
        sent_dict['text_heads'] = []

        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token['text'])
            sent_dict["pos"].append(token['upos'])
            head = token['head'] - 1 
            sent_dict['heads'].append(head)
            if head != -1:
                text_heads = spacy_token[head]['text']
                sent_dict['text_heads'].append(text_heads)
            else:
                sent_dict['text_heads'].append('ROOT')
            sent_dict['deps'].append(token['deprel'])
        # print(sent)
        # print(sent_dict["tokens"])
        try:
            sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
            sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
            assert len(sent_dict['heads']) == len(sent_dict['tokens'])
        except:
            print('Something was wrong!')
            return None
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        if sent_id == None:
            print("False to find sent_id")
            print(f'mydict: {my_dict}')
            print(f"event: {event_dict}")
            continue
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"], event_dict["end_char"])
        if not all(tok in  my_dict["event_dict"][event_id]["mention"] for tok in my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]):
            print(f'{my_dict["event_dict"][event_id]["mention"]} - \
            {my_dict["sentences"][sent_id]["tokens"][my_dict["event_dict"][event_id]["token_id"][0]: my_dict["event_dict"][event_id]["token_id"][-1] + 1]}')
            print(f'{my_dict["event_dict"][event_id]}  - {my_dict["sentences"][sent_id]}')
    
    return my_dict


if __name__ == '__main__':
    
    # my_dict = cat_xml_reader(dir_name="/home/daclai/DiffusECI/data/EventStoryLine/annotated_data/v1.5/", file_name="1/1_1ecbplus.xml.xml", intra=True, inter=True)
    # with open("1_1ecbplus.xml.xml.json", 'w') as f:
    #     json.dump(my_dict,f, indent=6)
    
    my_dict = meci_tsvx_reader(dir_name="../data/meci-dataset/meci-v0.1-public/causal-en/train/", file_name="aviation_accidents-week2-nhung-27753327_chunk_2.ann.tsvx",)
    with open("aviation_accidents-week2-nhung-27753327_chunk_2.ann.tsvx.json", 'w') as f:
        json.dump(my_dict,f, indent=6)
