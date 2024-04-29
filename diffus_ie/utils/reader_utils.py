from collections import defaultdict
import datetime
import re
from typing import Dict, List, Tuple
import torch
import networkx as nx
from transformers import AutoModel, AutoTokenizer


def tokenized_to_origin_span(text: str, token_list: List[str]):
    token_span = []
    pointer = 0
    for token in token_list:
        start = text.find(token, pointer)
        if start != -1:
            end = start + len(token)
            pointer = end
            token_span.append([start, end])
            assert text[start: end] == token, f"{token}-{text}"
        else:
            print(f"We can find the possiton of {token} \ntext: {text}")
            # token_span.append([-100, -100])
    return token_span


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC


def sent_id_lookup(my_dict, start_char, end_char = None):
    # print(f"my_dict: {my_dict}")
    # print(f"start: {start_char}")
    # print(f"end: {end_char}")
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']


def id_lookup(span_SENT, start_char, end_char):
    token_id = []
    nearest = 0
    dist = 100000
    char_range = set(range(start_char, end_char))
    for i, token_span in enumerate(span_SENT):
        if len(set(range(token_span[0], token_span[1])).intersection(char_range)) > 0:
            token_id.append(i)
        if abs(token_span[0]  - start_char) < dist:
            dist = abs(token_span[0]  - start_char)
            nearest = i
    if len(token_id) == 0: 
        print('Cannot find id of this token so that use the nearest token id!')
        token_id.append(nearest)
        # raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {} \n end_char: {}".format(span_SENT, start_char, end_char))

    return token_id


def find_sent_id(sentences: List[Dict], mention_span: List[int]):
    """
    Find sentence id of mention (ESL)
    """
    for sent in sentences:
        if mention_span[0] >= sent['sent_start_char'] and mention_span[1] <= sent['sent_end_char']:
            return sent['sent_id']
    print(f"Cannot find the sent id of mention {mention_span} in sentences {sentences}")
    return None


def get_mention_span(span: str) -> List[int]:
    span = [int(tok.strip()) for tok in span.split('_')]
    return span
        

def find_m_id(mention: List[int], eventdict:Dict):
    for m_id, ev in eventdict.items():
        # print(mention, ev['mention_span'])
        if mention == ev['mention_span']:
            return m_id
    
    return None


