"""
Get a bio, output a json of instances
of words in our word list

Conda environment: pipe
"""
import json
import spacy
from collections import defaultdict, Counter
import multiprocessing
from tqdm import tqdm
import string
from urllib.parse import urlsplit
import os
from glob import glob
import csv
import gzip
import random
import numpy as np
import pandas as pd
from blingfire import text_to_sentences, text_to_words, text_to_words_with_offsets
import emoji
        
def count_roles_initial(): 
    '''
    This was used for providing an initial inspection of the dataset. 
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    in_folder = os.path.join(base_folder, "persona_occur")
    counts = Counter()
    for f in os.listdir(in_folder): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                roles = d['personas'].keys()
                for r in roles: 
                    spans = d['personas'][r]
                    is_poss = False
                    for s in spans: 
                        start = s[0]
                        end = s[1]
                        deps = s[2]
                        rels = []
                        for child in deps: 
                            child_text = child[0]
                            child_rel = child[1]
                            rels.append(child_rel)
                        if 'poss' in rels: 
                            is_poss = True
                    if not is_poss: 
                        counts[r] += 1
    with open(os.path.join(base_folder, 'initial_role_counts.csv'), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for tup in counts.most_common(): 
            writer.writerow([tup[0], tup[1]])
        
def get_occs_by_last(): 
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    job_last = defaultdict(list)
    job2cat = defaultdict(list)
    job2occ = defaultdict(list)
    for cat in occ_hierarchy: 
        for occ in occ_hierarchy[cat]: 
            for job in occ_hierarchy[cat][occ]: 
                job_toks = tuple(text_to_words(job).split(' '))
                job_last[job_toks[-1]].append(job_toks)
                job2cat[job_toks].append(cat)
                job2occ[job_toks].append(occ)
    return job_last, job2cat, job2occ
        
def count_onet_occ(): 
    '''
    This just exact string-matches to mentions of
    onet occupations. 
    '''
    job_last, job2cat, job2occ = get_occs_by_last()
    
    about_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(about_path + "/**/*.json.gz", recursive=True)
    cat_counts = Counter()
    job_counts = Counter()
    occ_counts = Counter()
    for filename in tqdm(result): 
        short_name = filename.split('/')[-1].replace('.json.gz', '')
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                tokens = text_to_words(text).split(' ')
                jobs = set()
                occs = set()
                cats = set()
                for idx, tok in enumerate(tokens): 
                    if tok in job_last: 
                        candidates = job_last[tok]
                        for cand in candidates: 
                            cand_len = len(cand)
                            if idx - cand_len + 1 >= 0: 
                                if cand == tuple(tokens[idx-cand_len + 1:idx + 1]): 
                                    jobs.add(' '.join(cand))
                                    occs.update(job2occ[cand])
                                    cats.update(job2cat[cand])
                # count each job / occupation / family once per website
                job_counts.update(list(jobs))
                occ_counts.update(list(occs))
                cat_counts.update(list(cats))

    output_folder = '/home/lucyl/llm_social_identities/outputs/identity/job_strmatch_occur/'
    with open(os.path.join(output_folder, 'job_counts.json'), 'w') as outfile: 
        json.dump(job_counts, outfile)
        
    with open(os.path.join(output_folder, 'occ_counts.json'), 'w') as outfile: 
        json.dump(occ_counts, outfile)
        
    with open(os.path.join(output_folder, 'cat_counts.json'), 'w') as outfile: 
        json.dump(cat_counts, outfile)
        
def occ_role_overlap(): 
    '''
    This prints out commonly extracted roles not in the onet hierarchy 
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    roles = []
    with open(os.path.join(base_folder, 'role_counts.csv'), 'r') as csv_file:  
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            roles.append(row[0])
            if i > 100: 
                break
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'            
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    all_jobs = set()
    for cat in occ_hierarchy: 
        for occ in occ_hierarchy[cat]: 
            all_jobs.update(occ_hierarchy[cat][occ])
          
    missing_roles = []
    for role in roles:
        if role not in all_jobs: 
            missing_roles.append(role)
           
    print(missing_roles)
    
def cleanup_roles(): 
    '''
    Takes in the output of the RoBERTA classifier and joins
    together word pieces, e.g. if a piece of a word is tagged as
    the positive class, we retrieve the entire word and its start/end.
    
    Note that the entire retrieved word is not lowercased. 
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    in_folder = os.path.join(base_folder, 'extracted_roles')
    out_folder = os.path.join(base_folder, 'extracted_roles_clean')
    os.makedirs(out_folder, exist_ok=True)
    bio_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(bio_path + "/**/*.json.gz", recursive=True)
    
    for filename in tqdm(result):
        f = filename.split('/')[-1].replace('.json.gz', '.jsonl')
        if not os.path.exists(os.path.join(in_folder, f)): continue 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            reformatted_roles = defaultdict(dict) # {url : {sent: {start: (end, word)} }}
            for line in infile: 
                d = json.loads(line)
                roles = d['roles']
                if not roles: continue
                url = d['url']
                sents = defaultdict(dict)
                for r in roles: 
                    sents[r['sent_id']][r['start']] = (r['end'], r['word'])
                reformatted_roles[url] = sents
        
        outfile = open(os.path.join(out_folder, f), 'w')
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                if url not in reformatted_roles: continue
                    
                out = {}
                out['url'] = url
                res = []
                partial = []
                sents = text_to_sentences(text).split('\n')
                for sent_id, sent in enumerate(sents): 
                    if sent_id not in reformatted_roles[url]: continue
                    tokens_space, offsets = text_to_words_with_offsets(sent)
                    tokens = tokens_space.split(" ")
                    role_starts = set(reformatted_roles[url][sent_id].keys())
                    for i, tok in enumerate(tokens): 
                        is_role = False
                        is_hyphen = False
                        start, end = offsets[i]
                        tok_indices = set(range(start, end))
                        overlap = tok_indices & role_starts
                        role_indices = set()
                        pieces = []
                        if overlap: 
                            is_role = True
                            for role_start in overlap: 
                                role_end, word = reformatted_roles[url][sent_id][role_start]
                                pieces.append(word)
                                role_indices.update(range(role_start, role_end))
                            if tok_indices - role_indices: 
                                # partially tagged
                                partial.append({'sent_id': sent_id, 'start': start, 'end': end, 'word': tok, 'pieces': pieces})
                            elif tok == '-' and i-1 >= 0 and i + 1 < len(tokens):
                                    is_hyphen = True
                                    tok = tokens[i-1] + tok + tokens[i+1]
                                    start = start - len(tokens[i-1])
                                    end = end + len(tokens[i+1])
                                    partial.append({'sent_id': sent_id, 'start': start, 
                                                    'end': end, 'word': tok, 'pieces': ['-']})
                        if is_role: 
                            res.append({'sent_id': sent_id, 'start': start, 'end': end, 'word': tok})
                out['roles'] = res
                out['partial'] = partial # whole words that are only partially tagged
                outfile.write(json.dumps(out) + '\n')
        outfile.close()
        
def count_roles(): 
    '''
    Output all of the roles and how many times they appear in the dataset
    How many roles are in each website, mean + std? 
    How many websites (what percentage) have roles? 
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    in_folder = os.path.join(base_folder, 'extracted_roles_clean')
    
    total = 0
    total_websites = 0
    partial_count = 0
    counts = Counter()
    hyphen_pieces_count = Counter()
    num_roles = []
    for f in tqdm(os.listdir(in_folder)): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                num_r = len(d['roles'])
                if num_r > 0: 
                    total_websites += 1
                    num_roles.append(num_r)
                total += num_r
                if len(d['partial']) > 0: 
                    partial_count += len(d['partial'])
                for p in d['partial']: 
                    if p['pieces'] == ['-']: 
                        hyphen_pieces_count.update(p['word'].lower().split('-'))
                for role in d['roles']: 
                    w = role['word'].lower()
                    counts[w] += 1
                    
    print("Partial wordpieces tagged:", partial_count / total)
    print("Total websites with roles:", total_websites)
    print("Common hyphen components:", hyphen_pieces_count.most_common(100))
    print("Mean roles per website:", np.mean(num_roles), "standard deviation:", np.std(num_roles))

    with open(os.path.join(base_folder, 'role_counts.csv'), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for tup in counts.most_common(): 
            writer.writerow([tup[0], tup[1]])
            
def get_top_N(N=100): 
    '''
    Only inspect the top N roles
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    in_folder = os.path.join(base_folder, 'extracted_roles_clean')
    
    roles_to_keep = set()
    with open(os.path.join(base_folder, 'role_counts.csv'), 'r') as csv_file:  
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader): 
            role = row[0]
            roles_to_keep.add(role)
            if i == N: 
                break
                
    role_d = {
        'hn': [], 
        'role': []
    }
    for f in tqdm(os.listdir(in_folder)): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['url']
                u = urlsplit(url)
                hn = u.hostname
                role_set = set([role['word'].lower() for role in d['roles']])
                for w in role_set: 
                    if w in roles_to_keep: 
                        role_d['hn'].append(hn)
                        role_d['role'].append(w)
                        
    role_df = pd.DataFrame.from_dict(role_d)
    role_df.to_csv(os.path.join(base_folder, f'hn_top_{N}_roles.csv'))
    
def reverse_hierarchy(base_folder): 
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    job2occ = defaultdict(set)
    job2fam = defaultdict(set)
    for cat in occ_hierarchy: 
        for occ in occ_hierarchy[cat]: 
            jobs = occ_hierarchy[cat][occ]
            for job in jobs: 
                job2occ[job].add(occ)
                job2fam[job].add(cat)
        
    return job2occ, job2fam
    
def get_onet_jobs(): 
    '''
    This connects mentioned roles to ONET occupations
    (e.g. a designer, if prefaced by "interior designer" is also assigned
    that label)
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    data_folder = "/home/lucyl/llm_social_identities/data/person_lists/"
    in_folder = os.path.join(base_folder, 'extracted_roles_clean')
    out_folder = os.path.join(base_folder, 'extracted_roles_onet')
    os.makedirs(out_folder, exist_ok=True)
    bio_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(bio_path + "/**/*.json.gz", recursive=True)
    
    job2occ, job2fam = reverse_hierarchy(data_folder)
    
    with open(os.path.join(data_folder, 'last_to_job.json'), 'r') as infile: 
        last_to_job = json.load(infile)
        
    common_roles = set()
    with open(os.path.join(base_folder, 'role_counts.csv'), 'r') as csv_file:  
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            common_roles.add(row[0])
            if i > 500: 
                break
    
    for filename in tqdm(result):
        f = filename.split('/')[-1].replace('.json.gz', '.jsonl')
        if not os.path.exists(os.path.join(in_folder, f)): continue 
        url_to_roles = {}
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['url']
                sents = defaultdict(list)
                for r in d['roles']: 
                    sents[int(r['sent_id'])].append((r['start'], r['end'], r['word']))
                url_to_roles[url] = sents
        
        outfile = open(os.path.join(out_folder, f), 'w')
        # some hyphenated components of roles are not roles themselves
        common_nonrole_piece = set(['co', 'home', 'multi', 'in', 'something', 'vice', 'up', 'year', 'old', 'at',
                                'ordinator', 'all', 'ex', 'of', 'go', 'self', 'home', 'stand', 'to', 'end', 
                                    'past', 'er', 'sub', 'step', 'elect', 'turned', 'a', 'film', 'story', 'law', 
                                    'residence', 'twenty', 'large', 'web', 'post'])
        # output format: {url : url, 
        # roles : [onet roles with start / end / sent_id], 
        # occupations : [list of onet occ],  
        # job families : [list of job families]}
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                if url not in url_to_roles: continue
                    
                out = {}
                out['url'] = url
                res = []
                jobs = set()
                occs = set()
                fams = set()
                non_onet_additions = defaultdict(list)
                sents = text_to_sentences(text).split('\n')
                for sent_id, sent in enumerate(sents): 
                    if sent_id not in url_to_roles[url]: continue
                    for tup in url_to_roles[url][sent_id]: # for each role in this sentence
                        start, end, word = tup 
                        new_word = word.lower()
                        if new_word in common_nonrole_piece and (sent[end:end+1] == '-' or sent[end-1:end] == '-'): 
                            # avoid common parts of hyphenated words that aren't roles on their own
                            continue 
                        is_plural = False
                        if new_word.endswith('s') and (new_word[:-1] in last_to_job or new_word[:-1] in common_roles): 
                            # plurals
                            new_word = new_word[:-1]
                            end = end - 1
                            is_plural = True
                        if new_word not in last_to_job: 
                            # not an ONET occupation, but we still consider including
                            cand_len = len(new_word)
                            non_onet_additions[new_word].append((sent_id, end-cand_len, end))
                            #res.append((sent_id, end-cand_len, end, new_word)) 
                            continue
                        if new_word in last_to_job: 
                            for candidate in last_to_job[new_word]: 
                                candidate = candidate.lower()
                                cand_len = len(candidate)
                                if sent[end-cand_len:end].lower() == candidate:
                                    if candidate == 'is professor': 
                                        # ambiguity when case-insensitive
                                        if not sent[end-cand_len:end] == 'IS professor': continue
                                    res.append((sent_id, end-cand_len, end, candidate))
                                    if candidate not in jobs: 
                                        jobs.add(candidate)
                                        occs.update(job2occ[candidate])
                                        fams.update(job2fam[candidate])
                                elif sent[end-cand_len-1:end].lower().replace('-', '') == candidate: 
                                    # accounts for "make-up artist", "non-profit director", etc not appearing in ONET
                                    res.append((sent_id, end-cand_len-1, end, candidate))
                                    if candidate not in jobs: 
                                        jobs.add(candidate)
                                        occs.update(job2occ[candidate])
                                        fams.update(job2fam[candidate])
                                        
                occ_indices = set() # [(sent_id, index already accounted for by onet occ)]
                for tup in res: 
                    occ_indices.update([(tup[0], i) for i in range(tup[1], tup[2])])
                
                for word in non_onet_additions: 
                    for tup in non_onet_additions[word]: 
                        if (tup[0], tup[1]) not in occ_indices: 
                            # this extracted term is not part of an existing occupation
                            res.append((tup[0], tup[1], tup[2], word))

                out['job titles'] = res
                out['occupations'] = list(occs)
                out['job families'] = list(fams)
                outfile.write(json.dumps(out) + '\n')
        outfile.close()
        
def get_occupation_url_df(): 
    '''
    Creates the input for analysis and visualization of filtering. 
    There is an unusual case where "owner" is grouped by O*NET
    to be a urologist, but most owners in our dataset are not urologists but owners
    of businesses. 
    '''
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    data_folder = "/home/lucyl/llm_social_identities/data/person_lists/"
    job2occ, job2fam = reverse_hierarchy(data_folder)
    out2_d = {
        'hn': [], 
        'occupation': [],
    }
    ambiguity_cutoff = 1
    in_folder = os.path.join(base_folder, 'extracted_roles_onet')
    ambig_jobs = set()
    non_onet_roles = set()
    occ_job_counts = defaultdict(Counter)
    for f in tqdm(os.listdir(in_folder)): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                if len(d['job titles']) == 0: continue
                jobs = d['job titles']
                for tup in jobs: 
                    job = tup[3]
                    if len(job2occ[job]) > ambiguity_cutoff or job == 'owner': 
                        # ignore ambiguous job titles 
                        ambig_jobs.add(job)
                        continue
                    
                    for occ in job2occ[job]: 
                        occ_job_counts[occ][job] += 1
                    if len(job2occ[job]) == 0: # not in ONET
                        non_onet_roles.add(job)
    # rename occupations by top 3 most frequent job titles
    new_occ_name = {}
    for occ in occ_job_counts: 
        occ_name = ''
        for i, tup in enumerate(occ_job_counts[occ].most_common(3)): 
            occ_name += tup[0] + ', '
        new_occ_name[occ] = occ_name.strip()[:-1]
        
    # get prestige and salary for new occupation names
    prestige = {}
    salary = {}
    
    with open(os.path.join(data_folder, 'occ_prestige.json'), 'r') as infile: 
        occ_prestige = json.load(infile) # occupation as keys
    with open(os.path.join(data_folder, 'onet_salary.json'), 'r') as infile: 
        onet_salary = json.load(infile) # occ_hyph as keys
    with open(os.path.join(data_folder, 'job_salary.json'), 'r') as infile: 
        job_salary = json.load(infile) # job titles as keys
    with open(os.path.join(data_folder, 'job_prestige.json'), 'r') as infile: 
        job_prestige = json.load(infile) # job titles as keys
                
    for f in tqdm(os.listdir(in_folder)): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['url']
                u = urlsplit(url)
                hn = u.hostname
                jobs = d['job titles']
                occupations = set()
                for tup in jobs: 
                    job = tup[3]
                    if len(job2occ[job]) > ambiguity_cutoff or job == 'owner': 
                        occupations.add(job)
                        if job in job_salary: 
                            salary[job] = job_salary[job]
                        if job in job_prestige: 
                            prestige[job] = job_prestige[job]
                        continue
                    
                    for occ in job2occ[job]: 
                        occupations.add(new_occ_name[occ])
                        if new_occ_name[occ] not in prestige and occ in occ_prestige:
                            prestige[new_occ_name[occ]] = occ_prestige[occ]
                        if new_occ_name[occ] in salary: continue
                        occ_hyph = occ.translate(str.maketrans('', '', string.punctuation)).replace(' ', '-').lower()
                        if occ_hyph not in onet_salary: continue
                        sal = onet_salary[occ_hyph]
                        salary[new_occ_name[occ]] = sal
                        
                    if len(job2occ[job]) == 0: # not in ONET
                        occupations.add(job)

                for occ in occupations: 
                    out2_d['hn'].append(hn)
                    out2_d['occupation'].append(occ)
                    
    occ_metadata = {
        'occupation': [],
        'prestige': [], 
        'salary': []
    }
    for occ in set(out2_d['occupation']): 
        occ_metadata['occupation'].append(occ)
        occ_metadata['prestige'].append(prestige.get(occ, 0))
        occ_metadata['salary'].append(salary.get(occ, 0))
                    
    out2_df = pd.DataFrame.from_dict(out2_d)
    out2_df.to_csv(os.path.join(base_folder, 'hn_occupation.csv'))
    occ_metadata_df = pd.DataFrame.from_dict(occ_metadata)
    occ_metadata_df.to_csv(os.path.join(base_folder, 'newoccname_metadata.csv'))
    
def count_occ_families(): 
    base_folder = "/home/lucyl/llm_social_identities/outputs/identity/"
    in_folder = os.path.join(base_folder, 'extracted_roles_onet')
    fam_count = Counter()
    for f in tqdm(os.listdir(in_folder)): 
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                fam_count.update(d['job families'])
    print(fam_count.most_common())
        
if __name__ == "__main__":
    #cleanup_roles()
    #count_roles() 
    #get_onet_jobs()
    get_occupation_url_df()
    #count_occ_families()
            