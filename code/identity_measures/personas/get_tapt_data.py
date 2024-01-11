import pandas as pd
from tqdm import tqdm
import json
from glob import glob
import gzip
import os
import random

def get_individuals(): 
    pred_csv = 'about_pred.csv' # /home/lucyl/llm_social_identities/outputs/identity/about_pred.csv
    pred_indiv_org_df = pd.read_csv(pred_csv)
    individs = pred_indiv_org_df[pred_indiv_org_df['class'] == 0]
    urls = set(individs['url'].to_list())
    return urls

def write_data(): 
    '''
    awk "{if(rand()<0.9) {print > train.txt} else {print > val.txt}}" all.txt
    '''
    in_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    individ_urls = get_individuals()
    out_folder = '/home/lucyl/llm_social_identities/data/individuals'
    os.makedirs(out_folder, exist_ok=True)
    
    train_file = open(os.path.join(out_folder, 'train.txt'), 'w')
    val_file = open(os.path.join(out_folder, 'val.txt'), 'w')
    for filename in tqdm(result): 
        texts = []
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"].replace('\n', ' ').strip()
                url = row["id"]
                if url not in individ_urls: continue
                if random.random() < 0.1: 
                    val_file.write(text + '\n')
                else: 
                    train_file.write(text + '\n')
    train_file.close()
    val_file.close()
                    
if __name__ == "__main__":
    write_data()