"""
Scripts to calculate occupation-specific language
"""
import json
import spacy
from collections import defaultdict, Counter
from tqdm import tqdm
import re
import string
from urllib.parse import urlsplit
import os
import multiprocessing
from glob import glob
import csv
import gzip
import random
import math
import numpy as np
import pandas as pd
from blingfire import text_to_sentences, text_to_words, text_to_words_with_offsets
        
def process_batch(batch): 
    '''
    We count each word once per website to avoid
    repetition on websites from skewing the results too much
    '''
    short_name = batch['short_name']
    out_folder = batch['out_folder']
    filename = batch['filename']
    urls_to_count = batch['urls']
    hn_occ = batch['hn_occ']
    
    occ_to_word_counts = defaultdict(Counter)
    with gzip.open(filename, 'rt') as infile: 
        for line in infile: 
            row = json.loads(line)
            url = row['id']
            if url not in urls_to_count: continue
            u = urlsplit(url)
            hn = u.hostname
            toks = set(text_to_words(row["text"].lower()).split(' '))
            occ_to_word_counts['ALL'].update(toks)
            if hn in hn_occ: 
                for occ in hn_occ[hn]: 
                    occ_to_word_counts[occ].update(toks)
    with open(os.path.join(out_folder, short_name), 'w') as outfile: 
        json.dump(occ_to_word_counts, outfile)
        
def get_hn_occ(): 
    '''
    This matches `Webpages Visualized.ipynb`
    get occupations per hostname that occur at least 1k times, 
    and return a mapping from hostname to occupations
    '''
    base_folder = '/home/lucyl/llm_social_identities/outputs/identity/'
    hn_occ_df = pd.read_csv(os.path.join(base_folder, 'hn_occupation.csv'), index_col=0)
    occ_counts = hn_occ_df['occupation'].value_counts()
    hn_occ_df = hn_occ_df[hn_occ_df['occupation'].isin(occ_counts[occ_counts > 1000].index)]
    hn_occ_df = hn_occ_df[hn_occ_df['occupation'] != 'something'] # ambiguous age
    hn_occ_df = hn_occ_df[hn_occ_df['occupation'] != 'old'] # ambiguous age
    return hn_occ_df.groupby('hn')['occupation'].apply(list).to_dict()

def count_words(): 
    '''
    Count the word count for each hostname
    Get overall totals across all
    '''
    hn_occ = get_hn_occ()
    input_prefix = "/net/nfs/allennlp/lucyl/cc_data/cc_sample"
    result = glob(input_prefix + '/**/*.json.gz', recursive=True)
    
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)

    out_folder = "/home/lucyl/llm_social_identities/outputs/identity/word_counts/hn_counts"
    os.makedirs(out_folder, exist_ok=True)
    print('making batches...')
    batches = []
    for filename in tqdm(result): 
        urls = set(urls_per_basename[os.path.basename(filename)])
        b = {
            'filename': filename,
            'short_name': filename.split('/')[-1].replace('.json.gz', ''),
            'out_folder': out_folder,
            'urls': urls,
        } 
        b['hn_occ'] = defaultdict(list)
        for u in urls: 
            u_split = urlsplit(u)
            hn = u_split.hostname
            if hn not in hn_occ: continue
            b['hn_occ'][hn] = hn_occ[hn]
        batches.append(b)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
        list(tqdm(p.imap(process_batch, batches), total=len(batches)))

def agg_counts(): 
    '''
    Aggregate counts
    '''
    in_folder = "/home/lucyl/llm_social_identities/outputs/identity/word_counts"
    count_folder = os.path.join(in_folder, 'hn_counts')
    total_counts = Counter()
    occ_word_counts = defaultdict(Counter)
    d = defaultdict(dict)
    all_occs = set()
    for f in tqdm(os.listdir(count_folder)): 
        with open(os.path.join(count_folder, f), 'r') as infile: 
            this_occ_word_counts = json.load(infile) 
            for occ in this_occ_word_counts: 
                all_occs.add(occ)
                for w in this_occ_word_counts[occ]: 
                    if 'word' not in d[occ]: 
                        d[occ]['word'] = []
                    if 'count' not in d[occ]: 
                        d[occ]['count'] = []
                    d[occ]['word'].append(w)
                    d[occ]['count'].append(this_occ_word_counts[occ][w])
    
    for o in tqdm(all_occs): 
        this_df = pd.DataFrame.from_dict(d[o])
        word_counts = this_df.groupby(['word'], sort=False).sum().to_dict()['count']
        if o == 'ALL': 
            total_counts = word_counts
        else: 
            occ_word_counts[o] = word_counts
    
    with open(os.path.join(in_folder, 'total_counts.json'), 'w') as outfile: 
        json.dump(total_counts, outfile)
        
    with open(os.path.join(in_folder, 'occ_word_counts.json'), 'w') as outfile: 
        json.dump(occ_word_counts, outfile)
                        
def calculate_npmi(): 
    '''
    Calculate npmi for words that show up at least 20 times
    
    The npmi calculation if from Lucy et al. 2022 "Words as Gatekeepers"
    '''
    in_folder = "/home/lucyl/llm_social_identities/outputs/identity/word_counts"
    with open(os.path.join(in_folder, 'total_counts.json'), 'r') as infile: 
        total_counts = json.load(infile)
        
    with open(os.path.join(in_folder, 'occ_word_counts.json'), 'r') as infile: 
        occ_word_counts = json.load(infile)
                    
    overall_total = sum(total_counts.values())
    occ_npmi = defaultdict(dict)
    for occ in tqdm(occ_word_counts):
        word_counts = Counter(occ_word_counts[occ])
        total_j = sum(word_counts.values())
        pmi_d = {}
        for tup in word_counts.most_common(): 
            w = tup[0]
            c = tup[1]
            if c <= 20: 
                break # do not calculate npmi for rare words
            if not re.search('[a-z]', w): # word needs to contain a-z letters
                continue
            p_w_given_j = c / total_j
            p_w = total_counts[w] / overall_total
            pmi = math.log(p_w_given_j / p_w)
            h = -math.log(c / overall_total)
            pmi_d[w] = pmi / h
        occ_npmi[occ] = pmi_d    
    
    # save npmi scores
    out_folder = '/home/lucyl/llm_social_identities/outputs/identity/word_counts'
    with open(os.path.join(out_folder, 'occ_npmi.json'), 'w') as outfile: 
        json.dump(occ_npmi, outfile)
        
def density_helper(batch): 
    threshold = 0.1
    short_name = batch['short_name']
    out_folder = batch['out_folder']
    filename = batch['filename']
    urls_to_count = batch['urls']
    hn_occ = batch['hn_occ']
    occ_npmi = batch['occ_npmi']
    
    hn_prop = defaultdict(Counter) # {hn : {occ : prop} }
    with gzip.open(filename, 'rt') as infile: 
        for line in infile: 
            row = json.loads(line)
            url = row['id']
            if url not in urls_to_count: continue
            u = urlsplit(url)
            hn = u.hostname
            if hn not in hn_occ: continue
            
            toks = text_to_words(row["text"].lower()).split(' ')
            tok_counts = Counter(toks)
            num_toks = len(toks)
            occ_specific_wc = Counter()
            for tok in tok_counts: 
                for occ in hn_occ[hn]: 
                    if tok in occ_npmi[occ] and occ_npmi[occ][tok] > threshold: 
                        occ_specific_wc[occ] += 1
            for occ in hn_occ[hn]: 
                hn_prop[hn][occ] = occ_specific_wc[occ] / num_toks
    with open(os.path.join(out_folder, short_name), 'w') as outfile: 
        json.dump(hn_prop, outfile)

def get_jargon_density(): 
    '''
    For each hostname and each occupation, calculate the proportion of
    their webpage that uses occupation-specific language
    '''
    hn_occ = get_hn_occ()
    input_prefix = "/net/nfs/allennlp/lucyl/cc_data/cc_sample"
    result = glob(input_prefix + '/**/*.json.gz', recursive=True)
    
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
        
    in_folder = '/home/lucyl/llm_social_identities/outputs/identity/word_counts'
    with open(os.path.join(in_folder, 'occ_npmi.json'), 'r') as infile: 
        occ_npmi = json.load(infile)

    out_folder = "/home/lucyl/llm_social_identities/outputs/identity/word_counts/npmi_prop"
    os.makedirs(out_folder, exist_ok=True)
    print('making batches...')
    batches = []
    for filename in tqdm(result): 
        urls = set(urls_per_basename[os.path.basename(filename)])
        b = {
            'filename': filename,
            'short_name': filename.split('/')[-1].replace('.json.gz', ''),
            'out_folder': out_folder,
            'urls': urls,
            'occ_npmi': occ_npmi,
        } 
        b['hn_occ'] = defaultdict(list)
        for u in urls: 
            u_split = urlsplit(u)
            hn = u_split.hostname
            if hn not in hn_occ: continue
            b['hn_occ'][hn] = hn_occ[hn]
        batches.append(b)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 4) as p:
        list(tqdm(p.imap(density_helper, batches), total=len(batches)))
    pass

def more_jargon_than_average(): 
    '''
    Hostnames that use more occupation-specific language than average
    language for each occupation, create a dictionary of {occupation: {hn: top quartile True / False}}
    '''
    in_folder = "/home/lucyl/llm_social_identities/outputs/identity/word_counts"
    npmi_folder = os.path.join(in_folder, 'npmi_prop')

    occ_props = defaultdict(list)
    for f in tqdm(os.listdir(npmi_folder)): 
        with open(os.path.join(npmi_folder, f), 'r') as infile: 
            hn_prop = json.load(infile) 
        for hn in hn_prop: 
            for occ in hn_prop[hn]: 
                occ_props[occ].append(hn_prop[hn][occ])
                
    occ_avg_prop = Counter()
    for occ in occ_props: 
        occ_avg_prop[occ] = np.mean(occ_props[occ])
        
    hn_bucket = defaultdict(dict) 
    for f in tqdm(os.listdir(npmi_folder)): 
        with open(os.path.join(npmi_folder, f), 'r') as infile: 
            hn_prop = json.load(infile) 
        for hn in hn_prop: 
            for occ in hn_prop[hn]: 
                if hn_prop[hn][occ] > occ_avg_prop[occ]: 
                    hn_bucket[occ][hn] = True
                else: 
                    hn_bucket[occ][hn] = False
                    
    with open(os.path.join(in_folder, 'occ_hn_specific_bucket.json'), 'w') as outfile: 
        json.dump(hn_bucket, outfile)

if __name__ == "__main__":
    #count_words()
    #agg_counts()
    #calculate_npmi()
    #get_jargon_density()
    more_jargon_than_average()