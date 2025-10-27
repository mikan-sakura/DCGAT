#
#
# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import json
# import random
# from multiprocessing import Pool
# import functools
# import numpy as np
# from collections import defaultdict
# from itertools import chain
#
# from utils import Label2IdxSub, Label2IdxObj
#
#
# class InputExample(object):
#     """a single set of samples of data
#     """
#
#     def __init__(self, text, en_pair_list, re_list, rel2ens, tokens=None, pos_tags=None):
#         self.text = text
#         self.en_pair_list = en_pair_list
#         self.re_list = re_list
#         self.rel2ens = rel2ens
#         self.tokens = tokens  # 分词信息
#         self.pos_tags = pos_tags  # 词性标注信息
#
# class InputFeatures(object):
#     """
#     Desc:
#         a single set of features of data
#     """
#
#     def __init__(self,
#                  input_tokens,
#                  input_ids,
#                  attention_mask,
#                  seq_tag=None,
#                  corres_tag=None,
#                  relation=None,
#                  triples=None,
#                  rel_tag=None,
#                  pos_ids = None,
#                  bio_tags = None
#                  ):
#         self.input_tokens = input_tokens
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.seq_tag = seq_tag
#         self.corres_tag = corres_tag
#         self.relation = relation
#         self.triples = triples
#         self.rel_tag = rel_tag
#         self.pos_ids = pos_ids  # 词性标注ID
#         self.bio_tags = bio_tags  # BIO标签
#
# #cmeie
# def read_examples(data_dir, data_sign, rel2idx):
#     """load data to InputExamples
#     """
#     examples = []
#
#     # read src data
#     with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
#         data = json.load(f)
#         for sample in data:
#             text = sample['text']
#             tokens = sample['token']
#             pos_tags = sample['pos']
#             rel2ens = defaultdict(list)
#             en_pair_list = []
#             re_list = []
#
#             for triple in sample['triple_list']:
#                 en_pair_list.append([triple[0].lower(), triple[-1].lower()])
#                 re_list.append(rel2idx[triple[1]])
#                 rel2ens[rel2idx[triple[1]]].append((triple[0].lower(), triple[-1].lower()))
#             example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens,tokens=tokens,pos_tags=pos_tags)
#             examples.append(example)
#     print("InputExamples:", len(examples))
#     return examples
#
#
# #cmeie-v2
# # def read_examples(data_dir, data_sign, rel2idx):
# #     """Load data from new JSONL format into InputExamples."""
# #     examples = []
# #
# #     with open(data_dir / f'{data_sign}_triples.jsonl', "r", encoding="utf-8") as f:
# #         for line in f:
# #             sample = json.loads(line)
# #             text = sample["text"]
# #             rel2ens = defaultdict(list)
# #             en_pair_list = []
# #             re_list = []
# #
# #             for triple in sample["spo_list"]:
# #                 subject = triple["subject"].lower()
# #                 predicate = triple["predicate"]
# #                 obj = triple["object"]["@value"].lower()
# #
# #                 relation_id = rel2idx.get(predicate)
# #                 if relation_id is None:
# #                     continue  # skip unknown relations
# #
# #                 en_pair_list.append([subject, obj])
# #                 re_list.append(relation_id)
# #                 rel2ens[relation_id].append((subject, obj))
# #
# #             example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
# #             examples.append(example)
# #
# #     print("InputExamples:", len(examples))
# #     return examples
#
# def get_bio_labels(tokens):
#     """Generate BIO labels from token list
#     Args:
#         tokens: list of tokens
#     Returns:
#         list of BIO labels
#     """
#     labels = []
#     for token in tokens:
#         if len(token) == 1:
#             labels.append('S')  # Single character
#         else:
#             labels.extend(['B'] + ['I']*(len(token)-1))
#     return labels
#
# def find_head_idx(source, target):
#     target_len = len(target)
#     for i in range(len(source)):
#         if source[i: i + target_len] == target:
#             return i
#     return -1
#
#
# def _get_so_head(en_pair, tokenizer, text_tokens):
#     sub = tokenizer.tokenize(en_pair[0])
#     obj = tokenizer.tokenize(en_pair[1])
#     sub_head = find_head_idx(source=text_tokens, target=sub)
#     if sub == obj:
#         obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
#         if obj_head != -1:
#             obj_head += sub_head + len(sub)
#         else:
#             obj_head = sub_head
#     else:
#         obj_head = find_head_idx(source=text_tokens, target=obj)
#     return sub_head, obj_head, sub, obj
#
#
# def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params, pos_encoder, bio_encoder):
#     """convert function
#     """
#     text_tokens = tokenizer.tokenize(example.text)
#     # cut off
#     if len(text_tokens) > max_text_len:
#         text_tokens = text_tokens[:max_text_len]
#
#     # token to id
#     input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
#     attention_mask = [1] * len(input_ids)
#
#     # Process POS and BIO tags
#     bio_tags = get_bio_labels(example.tokens)
#
#     # Align POS and BIO tags with BERT tokenization
#     aligned_pos = []
#     aligned_bio = []
#     token_idx = 0
#
#     for token, pos, bio in zip(example.tokens, example.pos_tags, bio_tags):
#         sub_tokens = tokenizer.tokenize(token)
#         aligned_pos.extend([pos] * len(sub_tokens))
#         aligned_bio.extend([bio] * len(sub_tokens))
#
#         # Truncate to max length (accounting for [CLS] and [SEP])
#     aligned_pos = aligned_pos[:max_text_len - 2]
#     aligned_bio = aligned_bio[:max_text_len - 2]
#
#     # Add special tokens
#     aligned_pos = ['[CLS]'] + aligned_pos + ['[SEP]']
#     aligned_bio = ['[CLS]'] + aligned_bio + ['[SEP]']
#
#     # Pad to max length
#     if len(aligned_pos) < max_text_len:
#         aligned_pos += ['[PAD]'] * (max_text_len - len(aligned_pos))
#         aligned_bio += ['[PAD]'] * (max_text_len - len(aligned_bio))
#
#     # Convert to IDs
#     pos_ids = pos_encoder.transform(aligned_pos)
#     bio_ids = bio_encoder.transform(aligned_bio)
#
#     # zero-padding up to the sequence length
#     if len(input_ids) < max_text_len:
#         pad_len = max_text_len - len(input_ids)
#         # token_pad_id=0
#         input_ids += [0] * pad_len
#         attention_mask += [0] * pad_len
#
#     # train data
#     if data_sign == 'train':
#         # construct tags of correspondence and relation
#         corres_tag = np.zeros((max_text_len, max_text_len))
#         rel_tag = len(rel2idx) * [0]
#         for en_pair, rel in zip(example.en_pair_list, example.re_list):
#             # get sub and obj head
#             sub_head, obj_head, _, _ = _get_so_head(en_pair, tokenizer, text_tokens)
#             # construct relation tag
#             rel_tag[rel] = 1
#             if sub_head != -1 and obj_head != -1:
#                 corres_tag[sub_head][obj_head] = 1
#
#         sub_feats = []
#         # positive samples
#         for rel, en_ll in example.rel2ens.items():
#             # init
#             tags_sub = max_text_len * [Label2IdxSub['O']]
#             tags_obj = max_text_len * [Label2IdxSub['O']]
#             for en in en_ll:
#                 # get sub and obj head
#                 sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
#                 if sub_head != -1 and obj_head != -1:
#                     if sub_head + len(sub) <= max_text_len:
#                         tags_sub[sub_head] = Label2IdxSub['B-H']
#                         tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
#                     if obj_head + len(obj) <= max_text_len:
#                         tags_obj[obj_head] = Label2IdxObj['B-T']
#                         tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
#             seq_tag = [tags_sub, tags_obj]
#
#             # sanity check
#             assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
#                 attention_mask) == max_text_len, f'length is not equal!!'
#             sub_feats.append(InputFeatures(
#                 input_tokens=text_tokens,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 corres_tag=corres_tag,
#                 seq_tag=seq_tag,
#                 relation=rel,
#                 rel_tag=rel_tag,
#                 pos_ids=pos_ids,
#                 bio_tags=bio_ids
#             ))
#         # relation judgement ablation
#         if not ex_params['ensure_rel']:
#             # negative samples
#             neg_rels = set(rel2idx.values()).difference(set(example.re_list))
#             neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
#             for neg_rel in neg_rels:
#                 # init
#                 seq_tag = max_text_len * [Label2IdxSub['O']]
#                 # sanity check
#                 assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
#                 seq_tag = [seq_tag, seq_tag]
#                 sub_feats.append(InputFeatures(
#                     input_tokens=text_tokens,
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     corres_tag=corres_tag,
#                     seq_tag=seq_tag,
#                     relation=neg_rel,
#                     rel_tag=rel_tag,
#                     pos_ids=pos_ids,
#                     bio_tags=bio_ids
#                 ))
#     # val and test data
#     else:
#         triples = []
#         for rel, en in zip(example.re_list, example.en_pair_list):
#             # get sub and obj head
#             sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
#             if sub_head != -1 and obj_head != -1:
#                 h_chunk = ('H', sub_head, sub_head + len(sub))  #实体头开始位置，实体尾下一个位置
#                 t_chunk = ('T', obj_head, obj_head + len(obj))
#                 triples.append((h_chunk, t_chunk, rel)) #(('H', 7, 10), ('T', 76, 80), 33)
#         sub_feats = [
#             InputFeatures(
#                 input_tokens=text_tokens,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 triples=triples,
#                 pos_ids=pos_ids,
#                 bio_tags=bio_ids
#             )
#         ]
#
#     # get sub-feats
#     return sub_feats
#
#
# def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params, pos_encoder, bio_encoder):
#     """convert examples to features.
#     :param examples (List[InputExamples])
#     """
#     max_text_len = params.max_seq_length
#     # multi-process
#     with Pool(10) as p:
#         convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
#                                          data_sign=data_sign,pos_encoder=pos_encoder,bio_encoder=bio_encoder, ex_params=ex_params)
#         features = p.map(func=convert_func, iterable=examples)
#
#     return list(chain(*features))
#



# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain
from typing import List, Dict, Tuple

from utils import Label2IdxSub, Label2IdxObj


class InputExample(object):
    """a single set of samples of data"""

    def __init__(self, text, en_pair_list, re_list, rel2ens, tokens=None, pos_tags=None):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens
        self.tokens = tokens  # 分词信息
        self.pos_tags = pos_tags  # 词性标注信息


class InputFeatures(object):
    """a single set of features of data"""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None,
                 pos_ids=None,
                 bio_tags=None):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag
        self.pos_ids = pos_ids
        self.bio_tags = bio_tags

def _get_so_head(en_pair, tokenizer, text_tokens):
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:
        obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj

def clean_text(text: str) -> str:
    """Clean and normalize Chinese medical text"""
    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    text = text.replace('（', '(').replace('）', ')').replace('，', ',').replace('；', ';')
    return text.strip()


def find_head_idx(source: List[str], target: List[str]) -> int:
    """Improved entity head index finding with fuzzy matching"""
    target_len = len(target)
    if target_len == 0:
        return -1

    for i in range(len(source)):
        if source[i:i + target_len] == target:
            return i

    target_str = ''.join(target).replace('##', '')
    source_str = ''.join(source).replace('##', '')
    idx = source_str.find(target_str)
    if idx != -1:
        char_count = 0
        for token_idx, token in enumerate(source):
            char_count += len(token.replace('##', ''))
            if char_count > idx:
                return token_idx
    return -1


def get_bio_labels(tokens: List[str], entity_type: str = 'H') -> List[str]:
    """Generate BIO labels compatible with Label2IdxSub/Label2IdxObj"""
    return ['O'] * len(tokens)


def mark_entity(bio_tags: List[str], entity_tokens: List[str],
                all_tokens: List[str], entity_type: str) -> List[str]:
    """Mark entity positions in BIO tags"""
    start_idx = find_head_idx(all_tokens, entity_tokens)
    if start_idx != -1:
        end_idx = start_idx + len(entity_tokens)
        if end_idx <= len(bio_tags):
            bio_tags[start_idx] = f'B-{entity_type}'
            for i in range(start_idx + 1, end_idx):
                bio_tags[i] = f'I-{entity_type}'
    return bio_tags


def read_examples(data_dir, data_sign, rel2idx):
    """Enhanced data loading with better error handling"""
    examples = []

    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            try:
                text = clean_text(sample['text'])
                tokens = [clean_text(t) for t in sample['token']]
                pos_tags = sample['pos']

                if len(tokens) != len(pos_tags):
                    continue

                rel2ens = defaultdict(list)
                en_pair_list = []
                re_list = []

                for triple in sample['triple_list']:
                    try:
                        subject = clean_text(triple[0]).lower()
                        predicate = triple[1]
                        obj = clean_text(triple[2]).lower()

                        if not subject or not obj:
                            continue

                        relation_id = rel2idx.get(predicate)
                        if relation_id is None:
                            continue

                        en_pair_list.append([subject, obj])
                        re_list.append(relation_id)
                        rel2ens[relation_id].append((subject, obj))
                    except (IndexError, KeyError):
                        continue

                if en_pair_list:
                    example = InputExample(
                        text=text,
                        en_pair_list=en_pair_list,
                        re_list=re_list,
                        rel2ens=rel2ens,
                        tokens=tokens,
                        pos_tags=pos_tags
                    )
                    examples.append(example)
            except (KeyError, json.JSONDecodeError):
                continue

    print(f"Loaded {len(examples)} valid examples")
    return examples


def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params, pos_encoder, bio_encoder=None):
    """Enhanced feature conversion with better token alignment"""
    try:
        # Tokenize with BERT tokenizer
        text_tokens = tokenizer.tokenize(example.text)
        if len(text_tokens) > max_text_len - 2:
            text_tokens = text_tokens[:max_text_len - 2]

        # Convert to IDs
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
        attention_mask = [1] * len(input_ids)

        # Process POS tags
        aligned_pos = []
        for token, pos in zip(example.tokens, example.pos_tags):
            sub_tokens = tokenizer.tokenize(token)
            aligned_pos.extend([pos] * len(sub_tokens))
        aligned_pos = aligned_pos[:max_text_len - 2]
        aligned_pos = ['[CLS]'] + aligned_pos + ['[SEP]']
        if len(aligned_pos) < max_text_len:
            aligned_pos += ['[PAD]'] * (max_text_len - len(aligned_pos))
        pos_ids = pos_encoder.transform(aligned_pos)

        # Initialize BIO tags (all 'O')
        bert_bio_tags = ['O'] * len(text_tokens)

        # Mark entities
        for sub, obj in example.en_pair_list:
            sub_tokens = tokenizer.tokenize(sub)
            obj_tokens = tokenizer.tokenize(obj)
            bert_bio_tags = mark_entity(bert_bio_tags, sub_tokens, text_tokens, 'H')
            bert_bio_tags = mark_entity(bert_bio_tags, obj_tokens, text_tokens, 'T')

        # Add special tokens and pad
        bert_bio_tags = ['O'] + bert_bio_tags[:max_text_len - 2] + ['O']
        if len(bert_bio_tags) < max_text_len:
            bert_bio_tags += ['O'] * (max_text_len - len(bert_bio_tags))

        # Convert BIO tags to IDs (using direct mapping)
        bio_ids = [0] * max_text_len  # Default all to 'O'
        for i, tag in enumerate(bert_bio_tags):
            if tag == 'B-H' or tag == 'I-H':
                bio_ids[i] = Label2IdxSub[tag]
            elif tag == 'B-T' or tag == 'I-T':
                bio_ids[i] = Label2IdxObj[tag]

        # Pad input_ids and attention_mask
        if len(input_ids) < max_text_len:
            pad_len = max_text_len - len(input_ids)
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        # Training data processing
        if data_sign == 'train':
            corres_tag = np.zeros((max_text_len, max_text_len))
            rel_tag = len(rel2idx) * [0]

            for en_pair, rel in zip(example.en_pair_list, example.re_list):
                sub_head, obj_head, sub_tokens, obj_tokens = _get_so_head(
                    en_pair, tokenizer, text_tokens)
                rel_tag[rel] = 1

                if sub_head != -1 and obj_head != -1:
                    sub_end = min(sub_head + len(sub_tokens), max_text_len)
                    obj_end = min(obj_head + len(obj_tokens), max_text_len)
                    if sub_head < max_text_len and obj_head < max_text_len:
                        corres_tag[sub_head][obj_head] = 1

            sub_feats = []
            for rel, en_ll in example.rel2ens.items():
                tags_sub = [Label2IdxSub['O']] * max_text_len
                tags_obj = [Label2IdxObj['O']] * max_text_len

                for en in en_ll:
                    sub_head, obj_head, sub_tokens, obj_tokens = _get_so_head(
                        en, tokenizer, text_tokens)

                    if sub_head != -1 and sub_head < max_text_len:
                        sub_end = min(sub_head + len(sub_tokens), max_text_len)
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        for i in range(sub_head + 1, sub_end):
                            tags_sub[i] = Label2IdxSub['I-H']

                    if obj_head != -1 and obj_head < max_text_len:
                        obj_end = min(obj_head + len(obj_tokens), max_text_len)
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        for i in range(obj_head + 1, obj_end):
                            tags_obj[i] = Label2IdxObj['I-T']

                seq_tag = [tags_sub, tags_obj]

                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=rel,
                    rel_tag=rel_tag,
                    pos_ids=pos_ids,
                    bio_tags=bio_ids
                ))

            if not ex_params['ensure_rel']:
                neg_rels = set(rel2idx.values()).difference(set(example.re_list))
                neg_rels = random.sample(neg_rels, k=min(ex_params['num_negs'], len(neg_rels)))

                for neg_rel in neg_rels:
                    seq_tag = [[Label2IdxSub['O']] * max_text_len,
                               [Label2IdxObj['O']] * max_text_len]
                    sub_feats.append(InputFeatures(
                        input_tokens=text_tokens,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        corres_tag=corres_tag,
                        seq_tag=seq_tag,
                        relation=neg_rel,
                        rel_tag=rel_tag,
                        pos_ids=pos_ids,
                        bio_tags=bio_ids
                    ))
        else:
            triples = []
            for rel, en in zip(example.re_list, example.en_pair_list):
                sub_head, obj_head, sub_tokens, obj_tokens = _get_so_head(
                    en, tokenizer, text_tokens)

                if sub_head != -1 and obj_head != -1:
                    sub_end = min(sub_head + len(sub_tokens), max_text_len)
                    obj_end = min(obj_head + len(obj_tokens), max_text_len)
                    h_chunk = ('H', sub_head, sub_end)
                    t_chunk = ('T', obj_head, obj_end)
                    triples.append((h_chunk, t_chunk, rel))

            sub_feats = [InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples,
                pos_ids=pos_ids,
                bio_tags=bio_ids
            )]

        return sub_feats
    except Exception as e:
        print(f"Error processing example: {e}")
        return []


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params, pos_encoder,
                                 bio_encoder=None):
    """Convert examples to features with error handling"""
    max_text_len = params.max_seq_length

    with Pool(10) as p:
        convert_func = functools.partial(
            convert,
            max_text_len=max_text_len,
            tokenizer=tokenizer,
            rel2idx=rel2idx,
            data_sign=data_sign,
            ex_params=ex_params,
            pos_encoder=pos_encoder,
            bio_encoder=bio_encoder
        )
        features = []
        for result in p.imap(convert_func, examples):
            if result:
                features.extend(result)

    print(f"Converted {len(features)} features")
    return features