"""
A script for organizing training data for
various quality filters
"""
import os
import json
from collections import defaultdict
import gzip
import random
from glob import glob
from tqdm import tqdm
import numpy as np
import smart_open
from smashed.utils.io_utils import open_file_for_write, open_file_for_read, recursively_list_files
from datasets import load_dataset, get_dataset_config_names

def convert_combined_to_split(label_map, in_folder, out_folder): 
    '''
    This takes in a files that combine both classes and writes
    the data to two folders: one for the positive class, and
    one for the negative class. 
    
    The output files are .jsonl with one doc per line with 'text' key.
    '''
    for label in label_map: 
        dataset = label_map[label]
        os.makedirs(os.path.join(out_folder, dataset), exist_ok = True)
    for split in ['train', 'dev', 'test']: 
        data = defaultdict(list)
        with open(os.path.join(in_folder, split + '.jsonl'), 'r') as infile: 
            for line in infile: 
                row = json.loads(line)
                data[label_map[int(row['label'])]].append(row)

        for dataset in data: 
            with smart_open.open(os.path.join(out_folder, dataset, split + '.jsonl'), "wt") as outfile: 
                for row in data[dataset]: 
                    line = json.dumps(row) + '\n'
                    outfile.write(line)

def convert_split_to_combined(base_folder, data_name, neg_class): 
    '''
    Takes in an 'all.json' for a positive class
    and splits it into train/test/dev if they don't
    exist yet. Then, combines the negative and positive
    train/test/dev sets for each class (six files) into three total files. 
    '''
    if data_name == 'WikiWebBooks': return # we use Suchin's version for this
    if os.path.exists(os.path.join(base_folder, 'split', data_name, 'all.jsonl')): 
        if not (os.path.exists(os.path.join(base_folder, 'split', data_name, 'train.jsonl')) and \
                os.path.exists(os.path.join(base_folder, 'split', data_name, 'dev.jsonl')) and \
                os.path.exists(os.path.join(base_folder, 'split', data_name, 'test.jsonl'))): 
            idx = 0
            with open(os.path.join(base_folder, 'split', data_name, 'all.jsonl'), 'r') as infile: 
                for line in tqdm(infile): 
                    idx += 1
                    
            indices = list(range(idx))
            random.shuffle(indices)
            num_examples = len(indices) 
            cutoff = int(0.8*num_examples)
            train_idx = set(indices[:cutoff])
            other_idx = indices[cutoff:]
            cutoff = int(len(other_idx) / 2)
            dev_idx = set(other_idx[:cutoff])
            test_idx = set(other_idx[cutoff:])
            
            print(len(indices), len(train_idx), len(dev_idx), len(test_idx))
            
            train_file = open(os.path.join(base_folder, 'split', data_name, 'train.jsonl'), 'w')
            dev_file = open(os.path.join(base_folder, 'split', data_name, 'dev.jsonl'), 'w')
            test_file = open(os.path.join(base_folder, 'split', data_name, 'test.jsonl'), 'w')
            idx = 0
            with open(os.path.join(base_folder, 'split', data_name, 'all.jsonl'), 'r') as infile: 
                for line in tqdm(infile): 
                    if idx in train_idx: 
                        train_file.write(line.strip() + '\n')
                    elif idx in dev_idx: 
                        dev_file.write(line.strip() + '\n')
                    else: # test
                        test_file.write(line.strip() + '\n')
                    idx += 1
            train_file.close()
            dev_file.close()
            test_file.close()
    
    for split in ['train', 'dev', 'test']: 
        neg_path = os.path.join(base_folder, 'split', neg_class, split + '.jsonl')
        pos_path = os.path.join(base_folder, 'split', data_name, split + '.jsonl')
        out_path = os.path.join(base_folder, 'combined', data_name, split + '.jsonl')
        os.makedirs(os.path.join(base_folder, 'combined', data_name), exist_ok=True)
        os.system('cat ' + neg_path + ' ' + pos_path + ' > ' + out_path) 

def count_tokens_in_file(in_folder): 
    '''
    Counts number of white spaced tokens in each file in folder
    '''
    for f in os.listdir(in_folder): 
        print(in_folder, f)
        if not f.endswith('.jsonl'): continue
        total = 0
        with open(os.path.join(in_folder, f), 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                total += len(row['text'].split())
        print(f, total)
        
def gzip_open(file, mode, **open_kwargs):
    return gzip.open(filename=file, mode=mode, **open_kwargs)
        
def sample_from_wikipedia(base_folder, num_tokens=300000000): 
    '''
    num_tokens = number of white space tokens
    '''
    data_name = 'Wikipedia'
    source_path = 's3://ai2-llm/pretraining-data/sources/wikipedia/v0/documents/lang=en/'
    id_to_len = {}
    num_files = len(list(recursively_list_files(source_path)))
    for f in tqdm(recursively_list_files(source_path), total=num_files): 
        with open_file_for_read(f, "rb", open_fn=gzip_open) as infile: 
            for line in infile: 
                row = json.loads(line)
                text_len = len(row['text'].split())
                id_to_len[row['id']] = text_len
               
    keys = list(id_to_len.keys())
    random.seed(0)
    random.shuffle(keys)
    total = 0
    idx_to_keep = set()
    for idx in keys: 
        if total + id_to_len[idx] > num_tokens: 
            break
        total += id_to_len[idx]
        idx_to_keep.add(idx)
        
    out_folder = os.path.join(base_folder, 'split', data_name)
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, 'all.jsonl'), 'w') as outfile: 
        for f in tqdm(recursively_list_files(source_path), total=num_files): 
            with open_file_for_read(f, "rb", open_fn=gzip_open) as infile: 
                for line in infile: 
                    row = json.loads(line) 
                    if row['id'] not in idx_to_keep: continue
                    out = {}
                    out['id'] = row['id']
                    out['text'] = row['text']
                    out['label'] = 1
                    outfile.write(json.dumps(out) + '\n')
                    
def sample_from_wikiref(base_folder, num_tokens=300000000): 
    data_name = 'WikiRef'
    in_path = '/net/nfs.cirrascale/allennlp/lucyl/megawika_en'
    idx = 0
    id_to_len = {}
    for f in tqdm(sorted(os.listdir(in_path))): 
        with open(os.path.join(in_path, f), 'r') as infile: 
            for line in infile: 
                row = json.loads(line)
                for entry in row['entries']: 
                    text_len = len(entry['source_text'].split())
                    id_to_len[idx] = text_len
                    idx += 1
        
    keys = list(id_to_len.keys())
    random.seed(0)
    random.shuffle(keys)
    total = 0
    idx_to_keep = set()
    for idx in keys: 
        if total + id_to_len[idx] > num_tokens: 
            break
        total += id_to_len[idx]
        idx_to_keep.add(idx)
        
    out_folder = os.path.join(base_folder, 'split', data_name)
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, 'all.jsonl'), 'w') as outfile:
        idx = 0
        for f in tqdm(sorted(os.listdir(in_path))): 
            with open(os.path.join(in_path, f), 'r') as infile: 
                for line in infile: 
                    row = json.loads(line) 
                    for entry in row['entries']: 
                        if idx in idx_to_keep: 
                            out = {}
                            out['id'] = idx
                            out['text'] = entry['source_text']
                            out['label'] = 1
                            outfile.write(json.dumps(out) + '\n')
                        idx += 1
                        
def get_average_file_len(base_folder): 
    '''
    Gets the average white-space file length 
    of various datasets along with a few example shards of the CC
    we are using. 
    '''
    for folder in os.listdir(os.path.join(base_folder, 'split')): 
        if not os.path.exists(os.path.join(base_folder, 'split', folder, 'train.jsonl')): continue
        print(folder)
        lengths = []
        with open(os.path.join(base_folder, 'split', folder, 'train.jsonl'), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                lengths.append(len(d['text'].split()))
        print(np.mean(lengths), np.std(lengths))
        
    attribute_folder = '/home/lucyl/llm_social_identities/outputs/scores/whitespace_len/'
    lengths = []
    result = glob(attribute_folder + '/**/*.json.gz', recursive=True)
    for filename in tqdm(result): 
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                lengths.append(row['attributes']['cc__whitespace_tokenizer_v1__length'][0][2])
    print('Our CC:', np.mean(lengths), np.std(lengths))

if __name__ == '__main__':
#     count_tokens_in_file()
#     convert_split_to_combined()
    base_folder = '/home/lucyl/llm_social_identities/data/filter_data/'
    get_average_file_len(base_folder)
#     convert_combined_to_split({1: 'WikiWebBooks', 0: 'Random_CC'}, os.path.join(base_folder, 'combined/WikiWebBooks'), 
#                               os.path.join(base_folder, 'split'))
#     sample_from_wikipedia(base_folder)
#     sample_from_wikiref(base_folder)
#     count_tokens_in_file(os.path.join(base_folder, 'split/Wikipedia'))
#     count_tokens_in_file(os.path.join(base_folder, 'split/OpenWebText2'))
#     count_tokens_in_file(os.path.join(base_folder, 'split/WikiWebBooks'))
#     count_tokens_in_file(os.path.join(base_folder, 'split/Wikipedia'))
#     convert_split_to_combined(base_folder, 'WikiRef', 'Random_CC')
#     convert_split_to_combined(base_folder, 'Wikipedia', 'Random_CC')
#     convert_split_to_combined(base_folder, 'OpenWebText2', 'Random_CC')
#     count_tokens_in_file(os.path.join(base_folder, 'combined/WikiRef'))