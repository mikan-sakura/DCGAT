# DCGAT
Chinese Medical Relation Extraction Model Combining Dilated Convolution and Graph Attention

# Usage
1. Get pre-trained BERT model for PyTorch

Download chinese-roberta-wwm-ext which contains pytroch_model.bin, vocab.txt and config.json. Put these under ./pretrain_models.

2. Build Data

Put our preprocessed datasets under ./data.

3. Train

For example, to train the model for CMeIE dataset:

python train.py \
--ex_index=1 \
--epoch_num=100 \
--device_id=0 \
--corpus_type=CMeIE_pos \
--ensure_corres \
--ensure_rel

4. Evaluate

For example, to train the model for CMeIE dataset:

python evaluate.py \
--ex_index=1 \
--device_id=0 \
--mode=test \
--corpus_type=CMeIE_pos \
--ensure_corres \
--ensure_rel \
--corres_threshold=0.5 \
--rel_threshold=0.1

