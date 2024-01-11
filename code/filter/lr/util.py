"""
Copied from
https://raw.githubusercontent.com/kernelmachine/quality-filter/main/lr/util.py
"""

import os
import json
import numpy as np
import pandas as pd

def load_huggingface_tokenizer(tokenizer_path: str):
    with open(os.path.join(tokenizer_path, 'config.json'), 'r') as f:
            config = json.load(f)
    tokenizer_type = config['tokenizer_type']
    tokenizer = {'BPE': BPETokenizer,
                 'BBPE': ByteLevelBPETokenizer,
                 'BERT': BertWordPieceTokenizer}[tokenizer_type]
    if tokenizer_type in ['BPE', 'BBPE']:
        vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.json' in x][0]
        merges_file = [x for x in os.listdir(tokenizer_path) if 'merges.txt' in x][0]
        tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file),
                            merges_file=os.path.join(tokenizer_path, merges_file))
    else:
        vocab_file = [x for x in os.listdir(tokenizer_path) if 'vocab.txt' in x][0]
        tokenizer = tokenizer(vocab_file=os.path.join(tokenizer_path, vocab_file))
    return tokenizer


def jackknife(data, num_partitions=5):
    data = data.sample(frac=1)
    splits = np.split(data, range(0, data.shape[0], int(data.shape[0]/num_partitions) )[1:])
    for i, split in enumerate(splits):
        train_parts = list(range(0, num_partitions))
        try:
            train_parts.remove(i)
            yield pd.concat([splits[ix] for ix in train_parts], 0), split
        except ValueError:
            continue
    

def stratified_sample(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    rand_int = np.random.randint(1, 10000)
    df_ = df.groupby(col).apply(lambda x: x.sample(n, random_state=rand_int))
    df_.index = df_.index.droplevel(0)
    return df_


def replace_bool(x):
    if x == 'true':
        return 1
    elif x == 'false':
        return 0
    else:
        return x