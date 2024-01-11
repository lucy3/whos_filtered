"""
This script takes in a bunch of
retrieved about pages and gathers
root domains of these pages

Then, it grabs domains
that are not duplicated
throughout the dataset, prints
the count in an output file
and writes the urls to a list
"""
import os
from glob import glob
import json
import multiprocessing
from urllib.parse import urlparse
from tqdm import tqdm
from collections import Counter, defaultdict
import re
import pandas as pd
from bisect import bisect_left
import gzip
import ray
import random

OUTPUTS = '/home/lucyl/llm_social_identities/outputs'

def process_batch(batch):
    short_name = batch['short_name']
    out_folder = batch['out_folder']
    filename = batch['filename']
    hostnames = []
    keywords=set(['about-me', 'about', 'about-us', 'bio'])
    with gzip.open(filename, 'rt') as infile:
        for line in infile:
            row = json.loads(line)
            url = row['id']
            text = row['text']
            if len(text) < 100:
                # need to be long enough for a chance of reliable scores + measurement
                continue
            u = urlparse(url)
            hn = u.hostname
            p = u.path
            parts = list(filter(None, re.split("[/.]+", url)))
            if not keywords & set(parts):
                # about / bio needs to be in path, not hostname
                continue
            if not hn:
                print("Problem getting hostname from", url)
                continue
            hostnames.append((hn, url))
    with open(os.path.join(out_folder, short_name), 'w') as outfile:
        for tup in hostnames:
            outfile.write(tup[0] + '\t' + tup[1] + '\n')

def get_domains():
    '''
    Note that in the original version of cc_bios_v0, the files were labeled as ".gz" 
    but they were actually not zipped. In future runs
    of url_processor.py, they will be zipped. 
    '''
    input_prefix = "/net/nfs/allennlp/lucyl/cc_data/cc_bios_v0/"
    result = glob(input_prefix + '/**/*.json.gz', recursive=True)
    print('making batches...')

    out_folder = os.path.join(OUTPUTS, 'domains_with_about/splits/')
    os.makedirs(out_folder, exist_ok=True)
    batches = [{
        'filename': filename,
        'short_name': filename.split('/')[-1].replace('.json.gz', ''),
        'out_folder': out_folder,
    } for filename in result]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
        list(tqdm(p.imap(process_batch, batches), total=len(batches)))

def agg_domains():
    '''
    This creates two dictionaries: one of hostnames and the number of 
    about pages we retrieved for them, and one of hostnames 
    to a list of those pages. 
    '''
    hn_counts = Counter()
    hn_abouts = defaultdict(list)
    in_folder = os.path.join(OUTPUTS, 'domains_with_about/splits/')
    for f in tqdm(os.listdir(in_folder)):
        if not f.startswith('cc'): continue
        with open(os.path.join(in_folder, f), 'r') as infile:
            for line in infile:
                contents = line.strip().split('\t')
                hn_counts[contents[0]] += 1
                hn_abouts[contents[0]].append(contents[1])
                
    with open(os.path.join(OUTPUTS, 'domains_with_about/domain_counts.json'), 'w') as outfile:
        json.dump(hn_counts, outfile)
        
    with open(os.path.join(OUTPUTS, 'domains_with_about/domain_to_abouts.json'), 'w') as outfile:
        json.dump(hn_abouts, outfile)
        
def disambig_domains(): 
    '''
    Some webpages have multiple pages that have a path
    involving a target keyword. To narrow down through this ambiguity, we
    find the page that ends in /keyword/ or keyword.*, and if there
    is only one of these candidates, we map the hostname to that one. 
    We also account for differences in https vs http and we
    take the https version. 
    '''
    in_folder = os.path.join(OUTPUTS, 'domains_with_about/')
    with open(os.path.join(in_folder, 'domain_counts.json'), 'r') as infile:
        hn_counts = Counter(json.load(infile))
    with open(os.path.join(in_folder, 'domain_to_abouts.json'), 'r') as infile:
        hn_about = json.load(infile)
    keywords=set(['about-me', 'about', 'about-us', 'bio'])
    single_map = {}
    for hn in tqdm(hn_about): 
        match = None
        if hn_counts[hn] > 1: 
            candidates = []
            for url in hn_about[hn]: 
                # find urls that end in keyword as /about/ or about.html
                u = urlparse(url)
                parts = list(filter(None, u.path.split('/')))
                if not parts: continue
                parts = parts[-1]
                if '.' in parts: 
                    subparts = parts.split('.')
                    if len(subparts) == 2 and subparts[0] in keywords: 
                        candidates.append(url)
                else: 
                    if parts in keywords: 
                        candidates.append(url)
            if len(candidates) > 2 or len(candidates) == 0: 
                # too ambiguous
                continue 
            if len(candidates) == 1: 
                match = candidates[0]
            else: 
                assert len(candidates) == 2
                candidates = sorted(candidates)
                cand1 = urlparse(candidates[0])
                cand2 = urlparse(candidates[1])
                cand1_mod = cand1._replace(scheme="https")
                cand2_mod = cand2._replace(scheme="https")
                if cand1_mod == cand2_mod: 
                    match = candidates[1]
        else: 
            assert len(hn_about[hn]) == 1
            match = hn_about[hn][0]
        if match: 
            single_map[hn] = match
            
    with open(os.path.join(OUTPUTS, 'domains_with_about/domain_to_one_abouts.json'), 'w') as outfile:
        json.dump(single_map, outfile)

def domain_stats():
    in_folder = os.path.join(OUTPUTS, 'domains_with_about/')
    with open(os.path.join(in_folder, 'domain_counts.json'), 'r') as infile:
        d = Counter(json.load(infile))
        
    with open(os.path.join(in_folder, 'domain_to_one_abouts.json'), 'r') as infile:
        single_map = Counter(json.load(infile))

    keep_count = 0
    for k in tqdm(d):
        if d[k] == 1:
            keep_count += 1

    print("Writing stats...")
    with open(os.path.join(in_folder, 'domain_stats'), 'w') as outfile:
        total_hn = len(d)
        outfile.write("Total domains: {}\n".format(total_hn))
        outfile.write("Single page domains: {}\n".format(keep_count))
        outfile.write("Domains with the most possible about pages:\n")
        for tup in d.most_common(100):
            outfile.write('\t' + tup[0] + ' ' + str(tup[1]) + '\n')
        outfile.write("Found non-ambiguous about pages: {}\n".format(len(single_map)))
        
# def get_link_to_keep(batch): 
#     with open(batch['in_path'], 'r') as infile: 
#         with open(batch['out_path'], 'w') as outfile: 
#             for line in infile: 
#                 contents = line.strip().split('\t')
#                 hn = contents[0]
#                 url = contents[1]
#                 if url == batch['single_map'].get(hn): 
#                     outfile.write(url)

@ray.remote
def get_link_to_keep(single_map, batch):
    with open(batch['in_path'], 'r') as infile: 
        with open(batch['out_path'], 'w') as outfile: 
            for line in infile: 
                contents = line.strip().split('\t')
                hn = contents[0]
                url = contents[1]
                if url == single_map.get(hn): 
                    outfile.write(url + '\n')
    return 0
        
def to_keep_per_split(): 
    '''
    The out folder will contain 
    a list of about urls to keep per split. 
    This takes ~10 minutes
    '''
    in_folder = os.path.join(OUTPUTS, 'domains_with_about/splits/')
    out_folder = os.path.join(OUTPUTS, 'domains_with_about/to_keep_per_split/')
    
    with open(os.path.join(OUTPUTS, 'domains_with_about/domain_to_one_abouts.json'), 'r') as infile:
        single_map = Counter(json.load(infile))
        
    ray.init()
    data_id = ray.put(single_map)

    os.makedirs(out_folder, exist_ok=True)
    batches = [{
        'in_path': os.path.join(in_folder, filename),
        'out_path': os.path.join(out_folder, filename),
    } for filename in os.listdir(in_folder)]
    
    result_ids = []
    for i in batches:
        result_ids.append(get_link_to_keep.remote(data_id, i))

    # Get the results.
    results = ray.get(result_ids)
    
def count_jsons(filename): 
    with gzip.open(filename, 'rt') as infile:
        total = len(infile.readlines())
    return total

def sanity_check(): 
    '''
    Double check we got all of the pages. 
    '''
    input_prefix = "/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1/"
    result = glob(input_prefix + '/**/*.json.gz', recursive=True)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
        r = list(tqdm(p.imap(count_jsons, result), total=len(result)))
    
    print("Total number of websites:", sum(r))
    
def get_hostnames(): 
    with open(os.path.join(OUTPUTS, 'domains_with_about/domain_to_one_abouts.json'), 'r') as infile:
        single_map = Counter(json.load(infile))
    return set(single_map.keys())

@ray.remote
def filter_hostnames(hostnames, filename):
    filtered_counts = defaultdict(Counter) # { split : { hostname: # } }
    with gzip.open(filename, 'rt') as infile: 
        for line in infile: 
            d = json.loads(line.strip())
            break
        for hn in hostnames: 
            for split in d: # 'all' or 'long'
                if hn in d[split]: 
                    filtered_counts[split][hn] += d[split][hn]
        filtered_counts['all']['ALL'] = d['all']['ALL']
        filtered_counts['long']['ALL'] = d['long']['ALL']
    short_filename = filename.split('/')[-1].replace('.json.gz', '')
    return filtered_counts, short_filename
    
def agg_and_sample_webpage_counts(): 
    # This part takes ~12 mins 
    hostnames = get_hostnames()
    ray.init()
    data_id = ray.put(hostnames)
    
    input_prefix = "/net/nfs/allennlp/lucyl/cc_data/hostname_counts/"
    result = glob(input_prefix + '/**/*.json.gz', recursive=True)
    
    result_ids = []
    for i in result:
        result_ids.append(filter_hostnames.remote(data_id, i))
    
    results = ray.get(result_ids)
    
    print("Done getting results")
    
    # This part takes 40-ish minutes (worth the wait)
    print("Reservoir sampling")
    all_total = 0
    hn_total = 0 
    all_long_total = 0
    hn_long_total = 0
    k = 5
    reservoirs = defaultdict(list) # {hostname: list of filenames}
    hn_idx = Counter()
    random.seed(0)
    for res_tup in tqdm(results): 
        res, short_filename = res_tup
        all_long_total += res['long']['ALL']
        all_total += res['all']['ALL']
        hn_long_total += sum(res['long'].values()) - res['long']['ALL']
        hn_total += sum(res['all'].values()) - res['all']['ALL']
        
        # reservoir sample 5 not-to-short pages per hostname
        for hn in res['long']: 
            if hn == 'ALL': 
                continue 
            for i in range(res['long'][hn]): 
                # for the number of docs associated w/ this hostname in this split
                if hn_idx[hn] < k: 
                    assert len(reservoirs[hn]) < k
                    reservoirs[hn].append([short_filename, i]) # [name of file, ith json associated with hn in file]
                else: 
                    j = random.randrange(hn_idx[hn] + 1)
                    if j < k: 
                        reservoirs[hn][j] = [short_filename, i]
                hn_idx[hn] += 1 
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about/'
    with open(os.path.join(out_folder, 'hostname_totals.json'), 'w') as outfile: 
        json.dump(hn_total, outfile)
        
    with open(os.path.join(out_folder, 'sampled_webpage_loc.json'), 'w') as outfile: 
        json.dump(reservoirs, outfile)
        
    print("Writing stats...")
    with open(os.path.join(out_folder, 'domain_stats2'), 'w') as outfile:
        outfile.write("Num docs in CC_en: {}\n".format(all_total))
        outfile.write("Num not-too-short docs in CC_en: {}\n".format(all_long_total))
        outfile.write("Num docs with about pages: {}\n".format(hn_total))
        outfile.write("Num not-too-short docs with about pages: {}\n".format(hn_long_total))
        
def save_sample_size_per_split(): 
    '''
    Using the output of agg_and_sample_webpage_counts()
    for each split of CC we save a file containing
    the number of pages per hostname (if any) that we
    want to retrieve from that split. 
    Then, we reservoir sample through each split for
    those marked pages. 
    '''
    out_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about/'
    with open(os.path.join(out_folder, 'sampled_webpage_loc.json'), 'r') as infile: 
        reservoirs = json.load(infile)
        
    k = 5
    hostnames_per_split = defaultdict(dict) # { data_split : {hostnames: [idx in file] }} 
    num_hn = 0
    total_pages = 0
    print("Reversing map")
    for hn in tqdm(reservoirs): 
        num_hn += 1
        total_pages += len(reservoirs[hn])
        assert len(reservoirs[hn]) <= k
        for tup in reservoirs[hn]:
            filename = tup[0]
            i = tup[1]
            if hn in hostnames_per_split[filename]: 
                hostnames_per_split[filename][hn].append(i)
            else: 
                hostnames_per_split[filename][hn] = [i]
    
    print("Writing out reservoir")
    for filename in tqdm(hostnames_per_split): 
        with open(os.path.join(out_folder, 'reservoir', filename), 'w') as outfile: 
            json.dump(hostnames_per_split[filename], outfile)
        
    print("Writing stats...")
    with open(os.path.join(out_folder, 'domain_stats3'), 'w') as outfile:
        outfile.write("Num hostnames: {}\n".format(num_hn))
        outfile.write("Num pages in sample: {}\n".format(total_pages))
        
def phase1(): 
    '''
    This is run after running crawl_for_potential_about()
    in url_processor.py
    '''
    get_domains()
    agg_domains()
    disambig_domains()
    domain_stats()
    to_keep_per_split()
    
def phase2(): 
    '''
    This is run after running crawl_for_about_pages()
    in url_processor.py
    '''
    sanity_check()
    agg_and_sample_webpage_counts() 
    save_sample_size_per_split()

if __name__ == "__main__":
    #phase1()
    #phase2()
    domain_stats()
