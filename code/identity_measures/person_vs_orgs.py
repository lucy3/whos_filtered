"""
Measuring each about me page's
use of pronouns.
"""
import gzip
import json
import math
import multiprocessing
import os
import pdb
from collections import Counter, defaultdict
from glob import glob
from urllib.parse import urlsplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from blingfire import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import random
from joblib import dump, load
import pandas as pd
import csv
import numpy as np

def process_batch(batch):
    short_name = batch["short_name"]
    out_folder = batch["out_folder"]
    filename = batch["filename"]
    pronoun_to_series = batch["pronoun_to_series"]
    hostname_pronouns = defaultdict(Counter)
    with gzip.open(filename, "rt") as infile:
        for line in infile:
            row = json.loads(line)
            text = row["text"]
            url = row["id"]
            words = text_to_words(text).split(" ")
            hostname_pronouns[url]["total_words"] = len(words)
            for w in words:
                w = w.lower()
                if w in pronoun_to_series:
                    hostname_pronouns[url][w] += 1
    with open(os.path.join(out_folder, short_name + ".json"), "w") as outfile:
        json.dump(hostname_pronouns, outfile)
        
def get_pronoun_series(): 
    pronoun_series = {
        'i/me/my': set(["i", "me", "my", "mine", "myself"]),
        'we/us/our': set(["we", "us", "our", "ours"]),
    } 
    with open('/home/lucyl/llm_social_identities/data/pronouns', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t')
            key = '/'.join(contents[:3])
            pronoun_series[key] = set(contents)

    pronoun_to_series = defaultdict(set)
    for series in pronoun_series:
        for p in pronoun_series[series]:
            pronoun_to_series[p].add(series)
    return pronoun_series, pronoun_to_series

def count_pronouns(): 
    pronoun_series, pronoun_to_series = get_pronoun_series()

    in_path = "/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1"
    result = glob(in_path + "/**/*.json.gz", recursive=True)

    out_folder = "/home/lucyl/llm_social_identities/outputs/identity/pov/"
    os.makedirs(out_folder, exist_ok=True)

    batches = [
        {
            "filename": filename,
            "short_name": filename.split("/")[-1].replace(".json.gz", ""),
            "out_folder": out_folder,
            "pronoun_to_series": pronoun_to_series,
        }
        for filename in result
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
        list(tqdm(p.imap(process_batch, batches), total=len(batches)))
        
def get_pronoun_examples(): 
    '''
    Reservoir sample 50 examples of about pages containing 'they' and neopronouns
    '''
    output_folder = '/home/lucyl/llm_social_identities/outputs/identity/pov_examples/'
    
    pronoun_csv = '/home/lucyl/llm_social_identities/outputs/identity/url_pronoun.csv'
    
    reservoir = defaultdict(list) # { series : [examples] } 
    num_seen = Counter() # { series : count } 
    k = 50
    
    with open(pronoun_csv, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in tqdm(reader): 
            url = row['url']
            series = row['pronoun']
            if len(reservoir[series]) < k: 
                reservoir[series].append(url)
            else: 
                j = random.randrange(num_seen[series] + 1)
                if j < k: 
                    reservoir[series][j] = url
            num_seen[series] += 1
    
    url_to_series = {}
    for series in reservoir: 
        urls = reservoir[series]
        for url in urls: 
            url_to_series[url] = series
    
    in_path = "/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1"
    
    text_reservoir = defaultdict(list)
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    for filename in tqdm(result):        
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                if url not in url_to_series: continue
                    
                series = url_to_series[url]
                text_reservoir[series].append(text)
                
    for series in text_reservoir: 
        with open(os.path.join(output_folder, series.replace('/', '-') + '.txt'), 'w') as outfile: 
            for example in text_reservoir[series]: 
                outfile.write(example + '\n')
                outfile.write('\n-----------------------\n')
                
def get_most_pronoun_series(): 
    identity_folder = '/home/lucyl/llm_social_identities/outputs/identity/'
    pronoun_folder = os.path.join(identity_folder, 'pov')
    hn_top_pronoun = {
            'url': [], 
            'pronoun': [],
            'hn': [],
            'type': [],
        }
    pronoun_series, pronoun_to_series = get_pronoun_series()
    
    not_neopronouns = ['i/me/my', 'we/us/our', 'she/her/her', 'he/him/his', 'they/them/their']
    keywords=set(['about-me', 'about', 'about-us', 'bio'])
    
    for f in tqdm(os.listdir(pronoun_folder)): 
        with open(os.path.join(pronoun_folder, f), 'r') as infile: 
            d = json.load(infile)
        for url in d: 
            hn_counts = Counter(d[url]) # { pronoun : count } 
            series_counts = Counter()
            series_unique_count = Counter()
            for series in pronoun_series: 
                for p in pronoun_series[series]:
                    if p in hn_counts: 
                        series_unique_count[series] += 1
                        series_counts[series] = hn_counts[p]
            most_common_series = None
            for tup in series_counts.most_common(): 
                series, count = tup
                if series == 'it/it/its' or series == 'kit/kit/kits' or series =='total_words': continue
                if series not in not_neopronouns and series_unique_count[series] < 2: 
                    # often false positives for neopronouns
                    continue
                most_common_series = series
                break
            hn_top_pronoun['url'].append(url)
            u = urlsplit(url)
            hn = u.hostname
            hn_top_pronoun['hn'].append(hn)
            if most_common_series: 
                hn_top_pronoun['pronoun'].append(most_common_series)
            else: 
                hn_top_pronoun['pronoun'].append('none')
            parts = list(filter(None, re.split("[/.]+", url)))
            for k in keywords: 
                if k in parts: 
                    hn_top_pronoun['type'].append(k)
                    break
    hn_pronoun_df = pd.DataFrame.from_dict(hn_top_pronoun)
    hn_pronoun_df.to_csv(os.path.join(identity_folder, 'url_pronoun.csv'), index=False)
    
def get_person_features(row): 
    ents = row['ents']
    persons = set()
    person_count = 0
    for ent in ents: 
        if ent[1] == 'PERSON': 
            person_count += 1
            # person first token
            persons.add(ent[0].split()[0])
    uniq_person_count = len(persons)
    return person_count, uniq_person_count
    
def get_training_data(): 
    '''
    Get n random examples of about-us pages
    and n random examples of bio and about-me pages. 
    '''
    spacy_folder = '/home/lucyl/llm_social_identities/outputs/identity/spacy_output/'
    pronoun_csv = '/home/lucyl/llm_social_identities/outputs/identity/url_pronoun.csv'
    
    hn_pronoun_df = pd.read_csv(pronoun_csv)
    
    hn_pronoun_df.set_index('url',inplace=True)
    url_to_type = hn_pronoun_df.to_dict()['type']
    
    # reservoir sampling
    reservoir = defaultdict(list) # { class : [examples] } 
    num_seen = Counter() # { class : count } 
    k = 10000

    entity_files = os.listdir(spacy_folder)
    
    # ~30 seconds
    for f in tqdm(entity_files): 
        with open(os.path.join(spacy_folder, f), 'r') as infile: 
            short_name = f.split("/")[-1].replace(".json.gz", "")
            for line in infile: 
                row = json.loads(line)
                url = row['url']
                t = url_to_type[url]
                if t == 'bio' or t == 'about-me': 
                    clss = 'individual'
                elif t == 'about-us': 
                    clss = 'organization'
                else: 
                    continue

                if len(reservoir[clss]) < k: 
                    person_count, uniq_person_count = get_person_features(row)
                    reservoir[clss].append({'url': url, 'short_name': short_name, 'person_count': person_count, 'uniq_person_count': uniq_person_count})
                else: 
                    j = random.randrange(num_seen[clss] + 1)
                    if j < k: 
                        person_count, uniq_person_count = get_person_features(row)
                        reservoir[clss][j] = {'url': url, 'short_name': short_name, 'person_count': person_count, 'uniq_person_count': uniq_person_count}
                num_seen[clss] += 1
                
    url_to_keep = []
    for clss in reservoir: 
        url_dicts = reservoir[clss]
        for url_d in url_dicts: 
            url_to_keep.append(url_d['url'])
            
    print("Keeping", len(url_to_keep), "urls")
                
    pronoun_series, pronoun_to_series = get_pronoun_series()
    series_to_keep = ['i/me/my', 'we/us/our', 'she/her/her', 'he/him/his', 'they/them/their']
    pov_folder = "/home/lucyl/llm_social_identities/outputs/identity/pov/"
    url_to_pronouns = defaultdict(Counter)
    for f in tqdm(os.listdir(pov_folder)): 
        with open(os.path.join(pov_folder, f), 'r') as infile: 
            d = json.load(infile)
            for url in d: 
                if url not in url_to_keep: continue
                series_counts = Counter()
                hn_counts = d[url]
                
                for series in series_to_keep: 
                    for p in pronoun_series[series]:
                        if p in hn_counts: 
                            series_counts[series] = hn_counts[p]
                series_counts['total_words'] = hn_counts['total_words']
                url_to_pronouns[url] = series_counts
    
    url_to_keep = []
    for clss in reservoir: 
        url_dicts = reservoir[clss]
        for url_d in url_dicts: 
            pronoun_counts = url_to_pronouns[url_d['url']]
            total_word_count = pronoun_counts['total_words']
            url_to_keep.append({
                'url': url_d['url'],
                'person_count': url_d['person_count'] / total_word_count,
                'uniq_person_count': url_d['uniq_person_count'],
                'i/me/my': pronoun_counts['i/me/my']/ total_word_count,
                'we/us/our': pronoun_counts['we/us/our']/ total_word_count,
                'she/her/her': pronoun_counts['she/her/her']/ total_word_count,
                'he/him/his': pronoun_counts['he/him/his']/ total_word_count,
                'they/them/their': pronoun_counts['they/them/their']/ total_word_count,
                'class': clss, 
            })
    
    identity_folder = '/home/lucyl/llm_social_identities/outputs/identity/'
    
    with open(os.path.join(identity_folder, 'indiv_org_training.jsonl'), 'w') as outfile: 
        for example in url_to_keep: 
            outfile.write(json.dumps(example) + '\n')
    
def evaluate_classifier(): 
    '''
    5 fold cross validation of classifier
    '''
    identity_folder = '/home/lucyl/llm_social_identities/outputs/identity/'
    X = []
    y = []
    feature_order = ['person_count', 'uniq_person_count', 'i/me/my', 'we/us/our', 'she/her/her', 'he/him/his', 'they/them/their']
    with open(os.path.join(identity_folder, 'indiv_org_training.jsonl'), 'r') as infile: 
        for line in infile: 
            d = json.loads(line.strip())
            x = []
            for feat in feature_order: 
                x.append(d[feat])
            X.append(x)
            if d['class'] == 'individual': 
                y.append(0)
            else: 
                y.append(1)
    # hyperparameter tune
    grid_rf = {'n_estimators': [50, 100, 150, 200, 250, 300],
           'criterion': ['entropy', 'gini'], 
           'max_depth': [None, 10, 50, 70, 100],
           'min_samples_split': [2, 5, 10, 20],
           'min_samples_leaf': [1, 2, 4]}
    forest = RandomForestClassifier(random_state=0)
    r_rf = RandomizedSearchCV(forest, grid_rf, n_jobs=-1, scoring="f1_macro", random_state=0, n_iter=20)
    r_rf.fit(X, y)
    print('Best: %.3f' % r_rf.best_score_)
    print('\nBest params:\n', r_rf.best_params_)
    clf = r_rf.best_estimator_
    
    dump(clf, os.path.join(identity_folder, 'individ_org_model.joblib')) 
    
    probs = clf.predict_proba(X)
    true_positive = Counter()
    false_positive = Counter()
    for i in range(len(X)): 
        bucket_0 = math.ceil(probs[i][0]*10)/10 # rounded up to nearest 0.1
        if y[i] == clf.classes_[0]: 
            true_positive[bucket_0] += 1 
        else: 
            false_positive[bucket_0] += 1 
    
    x = []
    y = []
    z = []
    for bucket in sorted(true_positive.keys()): 
        x.append(bucket)
        y.append(true_positive[bucket])
        z.append(false_positive[bucket])
    fig, ax = plt.subplots(dpi=300)
    bottom = np.zeros(len(x))
    p = ax.bar(x, y, 0.1, label='y='+str(clf.classes_[0]), bottom=bottom)
    bottom += y
    p = ax.bar(x, z, 0.1, label='y='+str(clf.classes_[1]), bottom=bottom)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="True Label")
    ax.set_ylabel("Number of examples")
    ax.set_xlabel("Predicted class probability")
    plt.tight_layout()
    plt.savefig(os.path.join(identity_folder, 'individ_org_conf.png'))
    

def apply_classifier(): 
    '''
    This takes ~4 minutes to run. 
    '''
    identity_folder = '/home/lucyl/llm_social_identities/outputs/identity/'
    clf = load(os.path.join(identity_folder, 'individ_org_model.joblib'))
    
    spacy_folder = '/home/lucyl/llm_social_identities/outputs/identity/spacy_output/'
    pronoun_csv = '/home/lucyl/llm_social_identities/outputs/identity/url_pronoun.csv'
    pov_folder = "/home/lucyl/llm_social_identities/outputs/identity/pov/"
    
    print("Reading in types of urls...")
    hn_pronoun_df = pd.read_csv(pronoun_csv)
    hn_pronoun_df.set_index('url',inplace=True)
    url_to_type = hn_pronoun_df.to_dict()['type']

    entity_files = os.listdir(spacy_folder)
    
    print("Getting pronouns...")
    pronoun_series, pronoun_to_series = get_pronoun_series()
    series_to_keep = ['i/me/my', 'we/us/our', 'she/her/her', 'he/him/his', 'they/them/their']
    feature_order = ['person_count', 'uniq_person_count', 'i/me/my', 'we/us/our', 'she/her/her', 'he/him/his', 'they/them/their']
    
    labels = {
        'url': [], 
        'class': [],
    }
    
    for f in tqdm(entity_files): 
        short_name = f.split("/")[-1].replace(".ents", "")
        examples = []
        with open(os.path.join(spacy_folder, f), 'r') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['url']
                t = url_to_type[url]
                if t != 'about': 
                    labels['url'].append(url)
                    if t == 'about-us': 
                        labels['class'].append(1)
                    if t == 'about-me' or t == 'bio': 
                        labels['class'].append(0)
                    continue
                    
                person_count, uniq_person_count = get_person_features(row)
                examples.append({'url': url, 'short_name': short_name, 'person_count': person_count, 'uniq_person_count': uniq_person_count})
                
        if len(examples) == 0: continue
                
        url_to_pronouns = defaultdict(Counter)
        with open(os.path.join(pov_folder, short_name + '.json'), 'r') as infile: 
            d = json.load(infile)
            for url in d: 
                t = url_to_type[url]
                if t != 'about': continue

                series_counts = Counter()
                hn_counts = d[url]
                
                for series in series_to_keep: 
                    for p in pronoun_series[series]:
                        if p in hn_counts: 
                            series_counts[series] = hn_counts[p]
                series_counts['total_words'] = hn_counts['total_words']
                url_to_pronouns[url] = series_counts
                
        url_order = []
        X = []
        for url_d in examples: 
            pronoun_counts = url_to_pronouns[url_d['url']]
            total_word_count = pronoun_counts['total_words']
            x = []
            x.append(url_d['person_count'] / total_word_count) # person_count
            x.append(url_d['uniq_person_count']) # uniq_person_count
            x.append(pronoun_counts['i/me/my']/ total_word_count) # 'i/me/my'
            x.append(pronoun_counts['we/us/our']/ total_word_count)  # 'we/us/our'
            x.append(pronoun_counts['she/her/her']/ total_word_count)  # 'she/her/her'
            x.append(pronoun_counts['he/him/his']/ total_word_count)  # 'he/him/his'
            x.append(pronoun_counts['they/them/their']/ total_word_count)  # 'they/them/their'
            X.append(x)
            url_order.append(url_d['url'])
            
        y_pred = clf.predict(X)
        for i, url in enumerate(url_order): 
            labels['url'].append(url)
            labels['class'].append(y_pred[i])
    
    df = pd.DataFrame.from_dict(labels)
    df.to_csv(os.path.join(identity_folder, 'about_pred.csv'), index=False)    
            
def check_for_completeness(): 
    spacy_folder = '/home/lucyl/llm_social_identities/outputs/identity/spacy_output/'
    
    in_path = "/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1"
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    for filename in tqdm(result): 
        urls = set()
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                url = row["id"]
                urls.add(url)
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        found_urls = set()
        with open(os.path.join(spacy_folder, short_name + '.ents'), 'r') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['url']
                found_urls.add(url)
                
        if len(urls - found_urls) != 0: 
            print(short_name, len(urls - found_urls))
        
        
if __name__ == "__main__":
    #count_pronouns()
    #get_most_pronoun_series()
    #get_pronoun_examples()
    #check_for_completeness()
    #get_training_data()
    evaluate_classifier()
    #apply_classifier()
