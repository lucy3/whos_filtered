"""
Formatting for manual inspection
Take 100 random files in and then
take the first line and output what
we care about to a csv
"""
from glob import glob
import json
import os
import random
import csv
import gzip
from tqdm import tqdm
from urllib.parse import urlsplit
from collections import defaultdict, Counter
import numpy as np

ROOT = '/home/lucyl/llm_social_identities/'

def random_pages(): 
    '''
    Reservoir sample for 100 random pages and their scores
    Only returns two filter scores, though since this is an old function
    '''
    random.seed(0)

    in_path = '/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1'
    sample_files = glob(in_path + '/**/*.json.gz', recursive=True)

    in_folder = ROOT + 'outputs/scores/'
    with open(os.path.join(in_folder, 'ccnet_lang.json'), 'r') as infile: 
        lang_score = json.load(infile)
    with open(os.path.join(in_folder, 'ccnet_quality.json'), 'r') as infile: 
        qual_score = json.load(infile)

    reservoir = []
    num_seen = 0

    for f_path in tqdm(sample_files): 
        short_name = f_path.split('/')[-1].replace('.json.gz', '')
        with gzip.open(f_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['id']
                u = urlsplit(url)
                hn = u.hostname
                if num_seen < 100: 
                    reservoir.append({'file': short_name, 'id': d['id'], 'text': d['text'], 
                                'Website avg CCNet Wikipedia Perplexity': np.mean(qual_score[hn]), 
                                 'Website avg CCNet Language English ID': np.mean(lang_score[hn])})
                else: 
                    j = random.randrange(num_seen + 1)
                    if j < 100: 
                        reservoir[j] = {'file': short_name, 'id': d['id'], 'text': d['text'], 
                                'Website avg CCNet Wikipedia Perplexity': np.mean(qual_score[hn]), 
                                 'Website avg CCNet Language English ID': np.mean(lang_score[hn])}
                num_seen += 1

    with open(ROOT + 'outputs/domains_with_about/random_examples.csv', 'w') as outfile: 
        fieldnames = ['file', 'id', 'text', 'Website avg CCNet Wikipedia Perplexity', 'Website avg CCNet Language English ID']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for d in reservoir: 
            writer.writerow(d)
            
def filtered_topic_pages(): 
    '''
    This reservoir samples 20 pages per topic
    from a list of topics 
    '''
    cluster_folder = ROOT + 'outputs/kmeans/about_out/50/'
    cluster_num_to_terms = {}
    with open(ROOT + 'outputs/kmeans/about/50/top_terms.csv', 'r') as infile:
        for line in infile: 
            contents = line.strip().split(',')
            if contents[0] == '': continue
            cluster_num_to_terms[int(contents[0])] = ','.join(contents[1:4])
            
    clusters_of_interest = ['fashion,women,brand', 'online,store,shopping', 'company,products,quality',
                            'quality,equipment,production', 'products,quality,product', 'furniture,jewelry,quality',
                           'com,www,https']
    assert set(clusters_of_interest) - set(cluster_num_to_terms.values()) == set()
            
    cluster_hn = defaultdict(set) # {cluster to hostname}
    for f in tqdm(os.listdir(cluster_folder)): 
        if f.endswith('_done.txt'): continue
        with open(os.path.join(cluster_folder, f), 'r') as infile: 
            d = json.load(infile)
        for url in d: 
            u = urlsplit(url)
            hn = u.hostname
            cluster = cluster_num_to_terms[d[url]]
            if cluster in clusters_of_interest: 
                cluster_hn[cluster].add(hn)
                
    sample_cluster_hn = defaultdict(list)
    target_sample = set()
    for cluster in cluster_hn: 
        sample_cluster_hn[cluster] = random.sample(list(cluster_hn[cluster]), 20)
        target_sample.update(sample_cluster_hn[cluster])
        
    reservoir = {} # {hostname: (url, example)} 
    num_seen = Counter() 
        
    in_path = '/net/nfs/allennlp/lucyl/cc_data/cc_sample'
    sample_files = glob(in_path + '/**/*.json.gz', recursive=True)
    
    for f_path in tqdm(sample_files): 
        short_name = f_path.split('/')[-1].replace('.json.gz', '')
        with gzip.open(f_path, 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['id']
                u = urlsplit(url)
                hn = u.hostname
                if hn not in target_sample: continue
                if num_seen[hn] < 1: 
                    reservoir[hn] = (url, d['text'], d['metadata']['language_score'])
                else: 
                    j = random.randrange(num_seen[hn] + 1)
                    if j < 1: 
                        reservoir[hn] = (url, d['text'], d['metadata']['language_score'])
                num_seen[hn] += 1
    for clust in sample_cluster_hn: 
        with open(ROOT + 'outputs/identity/sample_topics/' + clust + '.jsonl', 'w') as outfile: 
            for hn in sample_cluster_hn[clust]: 
                url = reservoir[hn][0]
                text = reservoir[hn][1]
                lang_score = reservoir[hn][2]
                outfile.write(json.dumps({'url': url, 'text': text, 'language_score': lang_score}) + '\n')

filtered_topic_pages()
            
