"""
python spacy_pos.py --data-dir /net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1 --output-dir /home/lucyl/llm_social_identities/outputs/identity/persona_occur --split 4
"""

import spacy
from tqdm import tqdm
import gzip
import os
from glob import glob
import json
import argparse
from collections import defaultdict, Counter
import pandas as pd

def get_paths(result, split): 
    '''
    For splitting the data across GPUs
    '''
    files_to_keep = []
    with open('input_' + str(split), 'r') as infile: 
        for line in infile: 
            filename = line.strip()
            filename = filename.split('/')[-1].replace('.json.gz', '')
            files_to_keep.append(filename) 
    paths_to_keep = []
    for filename in result: 
        short_name = filename.split('/')[-1].replace('.json.gz', '')
        if short_name in files_to_keep: 
            paths_to_keep.append(filename)
    print("Keeping", len(paths_to_keep))
    return paths_to_keep

def get_person_list(): 
    '''
    Load the ngram-bucketed person list
    '''
    with open('ngram_buckets.json', 'r') as infile: # DATA + 'person_lists/ngram_buckets.json'
        d = json.load(infile)
    return d

def get_individuals(): 
    pred_csv = 'about_pred.csv' # /home/lucyl/llm_social_identities/outputs/identity/about_pred.csv
    pred_indiv_org_df = pd.read_csv(pred_csv)
    individs = pred_indiv_org_df[pred_indiv_org_df['class'] == 0]
    urls = set(individs['url'].to_list())
    return urls

def run_spacy(in_path, out_folder, split): 
    '''
    Gets terms that refer to people whose last tokens are nouns on a page
    and all words and their pos and idx that are dependent
    on that last noun. 
    
    Note that for unigrams, the start and end we ended up storing are character indices, 
    while for bigrams and trigrams the start and end are token indices. 
    '''
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    result = get_paths(result, split)
    individ_urls = get_individuals()

    os.makedirs(out_folder, exist_ok=True)
    buckets = get_person_list()
    
    nlp = spacy.load('en_core_web_trf') 
    spacy.require_gpu() 

    for filename in tqdm(result): 
        url_ids = []
        texts = []
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                if url not in individ_urls: continue
                url_ids.append(url)
                texts.append(text[:1000000]) # account for spacy max len
        docs = nlp.pipe(texts, disable=['lemmatizer', 'ner'])
        out_file = open(os.path.join(out_folder, short_name + '.jsonl'), 'w')
        for j, doc in enumerate(docs): 
            roles = defaultdict(list)
            for i, token in enumerate(doc): 
                # check up to three token window
                unigram = token
                if unigram.text.lower() in buckets['1'] and unigram.pos_ == 'NOUN': 
                    start = unigram.idx
                    end = unigram.idx + len(unigram.text) - 1
                    roles[unigram.text.lower()].append((start, end, [[child.text, child.dep_] for child in unigram.children]))
                bigram = doc[i:i+2]
                if bigram.text == unigram.text: continue # end of doc
                if bigram.text.lower() in buckets['2'] and bigram[-1].pos_ == 'NOUN': 
                    roles[bigram.text.lower()].append((bigram.start, bigram.end, [[child.text, child.dep_] for child in bigram[-1].children]))
                trigram = doc[i:i+3]
                if trigram.text == bigram.text: continue # end of doc
                if trigram.text.lower() in buckets['3'] and trigram[-1].pos_ == 'NOUN': 
                    roles[trigram.text.lower()].append((trigram.start, trigram.end, [[child.text, child.dep_] for child in trigram[-1].children]))
            out = {}
            out['url'] = url_ids[j]
            out['personas'] = roles
            out_file.write(json.dumps(out) + '\n')
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--split', required=True, type=int)

    print("** Make sure you are using the most updated version of about_pred.csv and ngram_buckets.json **")
    args = parser.parse_args()
    run_spacy(args.data_dir, args.output_dir, args.split)