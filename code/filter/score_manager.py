"""
This script contains
functions for managing filtering
scores
"""
import os
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
import gzip
import numpy as np
from urllib.parse import urlsplit

def get_ccnet_scores(): 
    '''
    In our data, CCNet scores are already
    calculated. This takes about ~30 min.
    '''
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
    
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    lang_score = {} # { hostname: score } 
    qual_score = {} # { hostname: score } 
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[os.path.basename(filename)])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                lang_score[hn] = row['metadata']['language_score']
                qual_score[hn] = row['metadata']['perplexity']
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    with open(os.path.join(out_folder, 'ccnet_lang.json'), 'w') as outfile: 
        json.dump(lang_score, outfile)
        
    with open(os.path.join(out_folder, 'ccnet_quality.json'), 'w') as outfile: 
        json.dump(qual_score, outfile)
        
def get_lengths(): 
    '''
    From CCNet: 
    - length: number of chars
    - nlines: number of lines
    '''
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
    
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    length_dict = defaultdict(dict) # { hostname: len } 
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[os.path.basename(filename)])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                length_dict[hn]['length'] = row['metadata']['length']
                length_dict[hn]['nlines'] = row['metadata']['nlines']
                length_dict[hn]['len_per_line'] = row['metadata']['length'] / float(row['metadata']['nlines'])
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    with open(os.path.join(out_folder, 'ccnet_len.json'), 'w') as outfile: 
        json.dump(length_dict, outfile)
        
def get_ccnet_about_scores(): 
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    lang_score = {} # { hostname: score } 
    qual_score = {} # { hostname: score } 
    for filename in tqdm(sorted(result)): 
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                u = urlsplit(url)
                hn = u.hostname
                lang_score[hn] = row['metadata']['language_score']
                qual_score[hn] = row['metadata']['perplexity']
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    with open(os.path.join(out_folder, 'ccnet_lang_about.json'), 'w') as outfile: 
        json.dump(lang_score, outfile)
        
    with open(os.path.join(out_folder, 'ccnet_quality_about.json'), 'w') as outfile: 
        json.dump(qual_score, outfile)
        
def get_score_percentiles(): 
    '''
    Get the score that is in the 50th percentile
    of each filter. This was used for log odds analysis (did not work, deprecated). 
    '''
    filter_names = ['ccnet_lang', 'ccnet_quality']
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    score_cutoff = {} 
    for n in filter_names: 
        all_s = []
        with open(os.path.join(out_folder, n + '.json'), 'r') as infile: 
            s = json.load(infile)
        for hn in tqdm(s): 
            s_list = s[hn]
            all_s.extend(s_list)
        score_cutoff[n] = np.percentile(all_s, 50)
    with open(os.path.join(out_folder, '50_percentile.json'), 'w') as outfile: 
        json.dump(score_cutoff, outfile)
        
def get_doc_language_scores(attribute_folder, output_name, metadata_name): 
    '''
    This gets language scores outputted by dolma and reformats it
    to the same format as ccnet scores. 
    
    Input line format: 
    {"id":about_page_url,"attributes":{metadata_name:[[0,3513,0.99]]},"source":"common-crawl"}
    '''
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
        
    lang_score = {} # { hostname: score } 
    result = glob(attribute_folder + '/**/*.json.gz', recursive=True)
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[basename])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                lang_score[hn] = row['attributes'][metadata_name][0][2]
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    with open(os.path.join(out_folder, output_name + '.json'), 'w') as outfile: 
        json.dump(lang_score, outfile)
        
def get_quality_scores(attr_name): 
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
        
    attribute_folder = "/home/lucyl/llm_social_identities/outputs/scores/" + attr_name
    qual_score = {} # { hostname: score } 
    result = glob(attribute_folder + '/*.jsonl', recursive=True)
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename).replace('.jsonl', '.json.gz')
        urls = set(urls_per_basename[basename])
        with open(filename, 'r') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                qual_score[hn] = row['attributes'][attr_name]
    
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    with open(os.path.join(out_folder, attr_name + '.json'), 'w') as outfile: 
        json.dump(qual_score, outfile)
        
def get_urls(): 
    '''
    Get one random page per hostname (use same seed as score manager and notebook)
    Group pages by dataset split
    '''
    cc_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    result = glob(cc_path + '/**/*.json.gz', recursive=True)
    hn_urls = defaultdict(list) # { hostname: [urls] } 
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                u = urlsplit(url)
                hn = u.hostname
                hn_urls[hn].append((url, basename))
                
    random.seed(0)
    urls_per_basename = defaultdict(list)
    for hn in tqdm(sorted(hn_urls.keys())): 
        url, basename = random.choice(hn_urls[hn])
        urls_per_basename[basename].append(url)
        
    outfolder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(outfolder, 'one_page_per_hn.json'), 'w') as outfile: 
        json.dump(urls_per_basename, outfile)

if __name__ == '__main__':
    #get_urls()
#     get_ccnet_scores() 
    get_lengths()
    #get_ccnet_about_scores()
    #get_score_percentiles()
    #get_doc_language_scores('/home/lucyl/llm_social_identities/outputs/scores/ft_wikiwebbooks_doc', 
    #                        'ft_wikiwebbooks', 'cc__wikiwebbooks_doc__pos') 
#     get_quality_scores('wikiwebbooks')
#     get_quality_scores('wikipedia')
#     get_quality_scores('wikiref')
#     get_quality_scores('openwebtext2')
#     get_doc_language_scores('/home/lucyl/llm_social_identities/outputs/scores/cld2_lang_doc', 
#                             'cld2_lang', 'cc__cld2_en_doc_v2__en') 
#     get_doc_language_scores('/home/lucyl/llm_social_identities/outputs/scores/cld3_lang_doc',
#                             'cld3_lang', 'cc__cld3_en_doc_v2__en') 
#     get_doc_language_scores('/home/lucyl/llm_social_identities/outputs/scores/langdetect_doc', 
#                             'langdetect', 'cc__langdetect_en_doc_v2__en') 
    