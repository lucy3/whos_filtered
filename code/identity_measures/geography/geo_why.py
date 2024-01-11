"""
An initial look at what differentiates 
very filtered versus not as filtered pages
in each subcontinent region
"""
import json
import os
from urllib.parse import urlsplit
from collections import defaultdict, Counter
import pandas as pd
import random
import gzip
from tqdm import tqdm
import math
from glob import glob
from blingfire import *

ROOT = '/home/lucyl/llm_social_identities/'

def get_region_hn(): 
    '''
    Get a dictionary of {hostname : subcontinental region}
    '''
    with open(os.path.join(ROOT, 'outputs/identity/url_to_country.json'), 'r') as infile: 
        hn_to_country = json.load(infile)
        
    country_df = pd.DataFrame(hn_to_country.items(), columns=['hn', 'country'])
    metadata_path = os.path.join(ROOT, 'data/countries/metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    country_df = pd.merge(country_df, metadata_df, left_on='country', right_on='Code', how="left", sort=False)
    country_df = country_df[['hn', 'Subregion']]
    region_hn = pd.Series(country_df.Subregion.values,index=country_df.hn).to_dict()
    
    return region_hn
        
def get_word_counts_by_score(): 
    score_folder = os.path.join(ROOT, 'outputs/scores')
    # load 50% cutoff (finish function in score_manage.py)
    with open(os.path.join(score_folder, '50_percentile.json'), 'r') as infile: 
        percentiles = json.load(infile)
        
    group_hn = get_region_hn()
    
    out_folder = os.path.join(ROOT, 'outputs/identity/log_odds/word_counts')
        
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    for filename in tqdm(result): 
        bucketed_counts = defaultdict(dict) # {group: { top_scorename : Counter(), bottom_scorename: Counter() }}
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                u = urlsplit(url)
                hn = u.hostname
                if hn not in group_hn: continue
                group = group_hn[hn]
                
                text = row['text'].lower()
                # note that blingfire doesn't work on non-English, e.g. chinese doesn't get tokenized...
                words = text_to_words(text).split(" ")
                buckets = []
                
                if row['metadata']['language_score'] > percentiles['ccnet_lang']: 
                    buckets.append('top_' + 'ccnet_lang')
                else: 
                    buckets.append('bottom_' + 'ccnet_lang')
                    
                if row['metadata']['perplexity'] > percentiles['ccnet_quality']: 
                    buckets.append('top_' + 'ccnet_quality')
                else: 
                    buckets.append('bottom_' + 'ccnet_quality')
                    
                for bucket in buckets: 
                    if bucket not in bucketed_counts[group]: 
                        bucketed_counts[group][bucket] = Counter()
                    bucketed_counts[group][bucket].update(words)
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        with open(os.path.join(out_folder, short_name + '.json'), 'w') as outfile: 
            json.dump(bucketed_counts, outfile)

def format_bayesequal_input(): 
    '''
    all 3 input files to bayesequal.py should be space-separated, two columns, frequency followed by word
    #1371056 the
    #923839 and
    #765263 i
    '''
    in_folder = os.path.join(ROOT, 'outputs/identity/log_odds/word_counts')
    out_folder = os.path.join(ROOT, 'outputs/identity/log_odds/word_counts_formatted')
    filter_names = ['ccnet_lang', 'ccnet_quality']
    
    metadata_path = os.path.join(ROOT, 'data/countries/metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    groups = set(metadata_df['Subregion'].unique().tolist())
    
    print(groups)
    
    for i, group in enumerate(groups): 
        print(group, i, 'out of', len(groups))
        total_counts = defaultdict(Counter)
        top_counts = defaultdict(Counter)
        bottom_counts = defaultdict(Counter)
        for filename in tqdm(os.listdir(in_folder)): 
            with open(os.path.join(in_folder, filename), 'r') as infile: 
                bucketed_counts = json.load(infile)
                if group not in bucketed_counts: continue
                for name in filter_names: 
                    top_counts[name].update(bucketed_counts[group].get('top_' + name, {}))
                    bottom_counts[name].update(bucketed_counts[group].get('bottom_' + name, {}))
                    total_counts[name].update(bucketed_counts[group].get('top_' + name, {}))
                    total_counts[name].update(bucketed_counts[group].get('bottom_' + name, {}))
        for name in total_counts: 
            with open(os.path.join(out_folder, group.replace(' ', '_') + '-' + name + '-top.txt'), 'w') as outfile: 
                for w in top_counts[name]: 
                    outfile.write(str(top_counts[name][w]) + ' ' + w + '\n')
            with open(os.path.join(out_folder, group.replace(' ', '_') + '-' + name + '-bottom.txt'), 'w') as outfile: 
                for w in bottom_counts[name]: 
                    outfile.write(str(bottom_counts[name][w]) + ' ' + w + '\n') 
            with open(os.path.join(out_folder, group.replace(' ', '_') + '-' + name + '-total.txt'), 'w') as outfile: 
                for w in total_counts[name]: 
                    outfile.write(str(total_counts[name][w]) + ' ' + w + '\n')
                    
def run_bayesequal(): 
    metadata_path = os.path.join(ROOT, 'data/countries/metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    groups = set(metadata_df['Subregion'].unique().tolist())
    filter_names = ['ccnet_lang', 'ccnet_quality']
    
    out_folder = os.path.join(ROOT, 'outputs/identity/log_odds/')
    in_folder = os.path.join(ROOT, 'outputs/identity/log_odds/word_counts_formatted')
    for group in tqdm(groups): 
        for name in filter_names: 
            bottom_path = os.path.join(in_folder, group.replace(' ', '_') + '-' + name + '-bottom.txt')
            top_path = os.path.join(in_folder, group.replace(' ', '_') + '-' + name + '-top.txt')
            total_path = os.path.join(in_folder, group.replace(' ', '_') + '-' + name + '-total.txt')
            out_path = os.path.join(out_folder, group.replace(' ', '_') + '-' + name + '.txt')
            os.system('python bayesequal.py -f ' + bottom_path + ' -s ' + top_path + ' -p ' + total_path + ' > ' + out_path)
            
def examine_multiling_helper(lang_id_name, span_min_len=10): 
    print(lang_id_name)
    group_hn = get_region_hn()
    score_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    in_path = os.path.join(score_folder, lang_id_name)
    result = glob(in_path + '/**/*.json.gz', recursive=True)
    d = {
        'hn': [], 
        'subregion': [], 
        'languages': [], 
        'lang_count': [], 
    }
    hn_to_pages = defaultdict(list)
    for filename in tqdm(result): 
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                u = urlsplit(url)
                hn = u.hostname
                if hn not in group_hn or (type(group_hn[hn]) == float and math.isnan(group_hn[hn])): continue
                attr = row['attributes']
                langs = set()
                # only count langs attached to long-enough spans
                for l in attr: 
                    lang = l.split('__')[-1]
                    if lang.startswith('not_'): continue
                    for ex in attr[l]: 
                        if ex[1] - ex[0] < span_min_len: 
                            continue
                        langs.add(lang)
                lang_count = len(langs)
                subregion = group_hn[hn]
                hn_to_pages[hn].append((subregion, langs, lang_count))
                
    random.seed(0)
    # represent each website by one random webpage
    for hn in hn_to_pages: 
        subregion, langs, lang_count = random.choice(hn_to_pages[hn])
        d['hn'].append(hn)
        d['subregion'].append(subregion)
        d['languages'].append(langs)
        d['lang_count'].append(lang_count)
                
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(score_folder, lang_id_name + '__subregion.csv'))
    
def sample_per_subregion_lang_helper(lang_id_name, k=100, span_min_len=10): 
    group_hn = get_region_hn()
    score_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    cc_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_sample'
    in_path = os.path.join(score_folder, lang_id_name)
    result = glob(in_path + '/**/*.json.gz', recursive=True)
    
    reservoir = defaultdict(dict)
    reservoir_counter = Counter()
    for filename in tqdm(result): 
        shorter_name = filename.split('/')[-1]
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                u = urlsplit(url)
                hn = u.hostname
                if hn not in group_hn or (type(group_hn[hn]) == float and math.isnan(group_hn[hn])): continue
                subregion = group_hn[hn]
                attr = row['attributes']
                for l in attr: 
                    lang = l.split('__')[-1]
                    if lang.startswith('not_'): continue
                    key = subregion + ' - ' + lang
                    for ex in attr[l]: 
                        if ex[1] - ex[0] < span_min_len: 
                            continue
                        # decide whether to sample or not
                        if lang not in reservoir[subregion]: 
                            reservoir[subregion][lang] = []
                        if len(reservoir[subregion][lang]) < k: 
                            assert reservoir_counter[key] < k
                            reservoir[subregion][lang].append((shorter_name, url, ex[0], ex[1]))
                        else: 
                            j = random.randrange(reservoir_counter[key] + 1)
                            if j < k: 
                                reservoir[subregion][lang][j] = (shorter_name, url, ex[0], ex[1])
                        reservoir_counter[key] += 1 
                      
    # reformat so that key is filename split then url
    url_pool = defaultdict(dict)
    for subregion in reservoir: 
        for label in reservoir[subregion]: 
            assert len(reservoir[subregion][label]) <= k
            for tup in reservoir[subregion][label]: 
                if tup[1] not in url_pool[tup[0]]: 
                    url_pool[tup[0]][tup[1]] = []
                url_pool[tup[0]][tup[1]].append((subregion, label, tup[2], tup[3]))
                
    examples = defaultdict(dict)
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    for filename in tqdm(result): 
        shorter_name = filename.split('/')[-1]
        this_url_pool = url_pool[shorter_name]
        if len(this_url_pool) == 0:
            continue
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in this_url_pool: continue
                text = row['text']
                for tup in this_url_pool[url]: 
                    subregion, label, start, end = tup
                    span = text[start:end]
                    if label not in examples[subregion]: 
                        examples[subregion][label] = []
                    examples[subregion][label].append(span)
                    
    with open(os.path.join(score_folder, lang_id_name + '__examples.json'), 'w') as outfile: 
        json.dump(examples, outfile)

def examine_multiling(): 
    # call the helper function for all four language id at sentence and paragraph level 
    examine_multiling_helper('ft_lang_sent')
    examine_multiling_helper('ft_lang_para')
    
    examine_multiling_helper('cld2_lang_sent')
    examine_multiling_helper('cld2_lang_para')
    
    examine_multiling_helper('cld3_lang_sent')
    examine_multiling_helper('cld3_lang_para')
    
    examine_multiling_helper('langdetect_sent')
    examine_multiling_helper('langdetect_para')
    
def sample_per_subregion_lang(): 
    sample_per_subregion_lang_helper('ft_lang_sent')
    sample_per_subregion_lang_helper('ft_lang_para')
    
    sample_per_subregion_lang_helper('cld2_lang_sent')
    sample_per_subregion_lang_helper('cld2_lang_para')
    
    sample_per_subregion_lang_helper('cld3_lang_sent')
    sample_per_subregion_lang_helper('cld3_lang_para')
    
    sample_per_subregion_lang_helper('langdetect_sent')
    sample_per_subregion_lang_helper('langdetect_para')

if __name__ == "__main__":
    #get_word_counts_by_score()
    #format_bayesequal_input()
    #run_bayesequal()
    examine_multiling()
    sample_per_subregion_lang()