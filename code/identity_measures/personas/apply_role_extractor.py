"""
This script takes in a split of data, and 
then applies a roberta checkpoint onto that data. 
"""
import os
import json
import numpy as np
from collections import defaultdict
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import gzip
from glob import glob
from blingfire import text_to_sentences
import argparse

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

def get_individuals(): 
    pred_csv = 'about_pred.csv' # /home/lucyl/llm_social_identities/outputs/identity/about_pred.csv
    pred_indiv_org_df = pd.read_csv(pred_csv)
    individs = pred_indiv_org_df[pred_indiv_org_df['class'] == 0]
    urls = set(individs['url'].to_list())
    return urls

class NpEncoder(json.JSONEncoder):
    '''
    https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def run_roberta(in_path, out_folder, split): 
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    result = get_paths(result, split)
    individ_urls = get_individuals()
    
    os.makedirs(out_folder, exist_ok=True)
    
    model_path = 'model_path' 
    classifier = pipeline("ner", model=model_path)
    for filename in tqdm(result): 
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        out_file = open(os.path.join(out_folder, short_name + '.jsonl'), 'w')
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                if url not in individ_urls: continue
                out = {}
                out['url'] = url
                this_roles = []
                sents = text_to_sentences(text).split('\n')
                for i, res in enumerate(classifier(sents)): 
                    for tok in res: 
                        if tok['entity'] == 'LABEL_1': 
                            tok['sent_id'] = i
                            this_roles.append(tok)
                out['roles'] = this_roles
                out_file.write(json.dumps(out, cls=NpEncoder) + '\n')
        out_file.close()
                        
if __name__ == "__main__":
    '''
    example usage: 
    python apply_role_extractor.py --data-dir /net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1 --output-dir /home/lucyl/llm_social_identities/outputs/identity/extracted_roles --split 0
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--split', required=True, type=int)

    print("** Make sure you are using the most updated version of about_pred.csv and model_path **")
    args = parser.parse_args()
    run_roberta(args.data_dir, args.output_dir, args.split)
                    