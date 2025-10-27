
# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features

from sklearn.preprocessing import LabelEncoder


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=False)
        self.data_cache = params.data_cache

        # Initialize POS and BIO encoders
        self.pos_encoder = None
        self.bio_encoder = None
        self._init_encoders()

    def _init_encoders(self):
        """Initialize POS and BIO encoders using training data"""
        # Load training data to collect all POS tags
        with open(self.data_dir / 'rel2id.json', 'r', encoding='utf-8') as f_re:
            rel2idx = json.load(f_re)[-1]

        train_examples = read_examples(self.data_dir, data_sign='train', rel2idx=rel2idx)

        # Collect all POS tags
        pos_tags = set()
        for example in train_examples:
            for tag in example.pos_tags:
                pos_tags.add(tag)

        # Initialize encoders
        self.pos_encoder = LabelEncoder()
        self.pos_encoder.fit(list(pos_tags) + ['[CLS]', '[SEP]', '[PAD]'])

        self.bio_encoder = LabelEncoder()
        self.bio_encoder.fit(['B', 'I', 'S', '[CLS]', '[SEP]', '[PAD]'])

    @staticmethod
    def collate_fn_train(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) #(64,100)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long) #(64,100)
        seq_tags = torch.tensor([f.seq_tag for f in features], dtype=torch.long) #(64,2,100)
        poten_relations = torch.tensor([f.relation for f in features], dtype=torch.long) #(64,)
        corres_tags = torch.tensor([f.corres_tag for f in features], dtype=torch.long) #(64,100,100)
        rel_tags = torch.tensor([f.rel_tag for f in features], dtype=torch.long) #(64,53)
        pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        bio_tags = torch.tensor([f.bio_tags for f in features], dtype=torch.long)
        tensors = [input_ids, attention_mask, seq_tags, poten_relations, corres_tags, rel_tags,pos_ids, bio_tags]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        bio_tags = torch.tensor([f.bio_tags for f in features], dtype=torch.long)
        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [input_ids, attention_mask, triples, input_tokens,pos_ids, bio_tags,]
        return tensors

    def get_features(self, data_sign, ex_params):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # get relation to idx
            with open(self.data_dir / f'rel2id.json', 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1]
            # get examples
            if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
                examples = read_examples(self.data_dir, data_sign=data_sign, rel2idx=rel2idx) #包含en_pair_list，rel_list,rel2ens,text,是list集合
            else:
                raise ValueError("please notice that the data can only be train/val/test!!")
            features = convert_examples_to_features(self.params, examples, self.tokenizer, rel2idx, data_sign,
                                                    ex_params, pos_encoder=self.pos_encoder, bio_encoder=self.bio_encoder)
            # save data
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign="train", ex_params=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign, ex_params=ex_params)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params

    params = Params(corpus_type='WebNLG')
    ex_params = {
        'ensure_relpre': True
    }
    dataloader = CustomDataLoader(params)
    feats = dataloader.get_features(ex_params=ex_params, data_sign='test')
    print(feats[7].input_tokens)
