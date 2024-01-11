"""
Basic statistics about the dataset: 
- number of hostnames
- number of white-spaced tokens
"""
import os
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
import gzip
import numpy as np
from urllib.parse import urlsplit
import random

def get_token_count(): 
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
    
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    sample_len_total = 0
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[os.path.basename(filename)])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                tok_len = len(row["text"].split(' '))
                u = urlsplit(url)
                hn = u.hostname
                sample_len_total += tok_len
                
    print("Total sample white-space tokens:", sample_len_total)
                
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    about_len_total = 0
    for filename in tqdm(sorted(result)): 
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                tok_len = len(row["text"].split(' '))
                about_len_total += tok_len
    
    print("Total about white-space tokens:", about_len_total)

if __name__ == "__main__":
    get_token_count()