'''
Usage:
python locations.py --ent-dir /home/lucyl/llm_social_identities/outputs/identity/spacy_output --output-dir /home/lucyl/llm_social_identities/outputs/identity/geoparse --split 0 --data-dir /net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1 --host allennlp-cirrascale-01.reviz.ai2.in --port 9200

Locally this uses the geograph environment which has my fork of mordecai3
'''

import os
from tqdm import tqdm
import json
from mordecai3 import Geoparser
from collections import defaultdict, Counter
import argparse
from blingfire import text_to_sentences
import time
from glob import glob
import gzip
import logging
import random

logging.disable(logging.INFO)

def get_paths(result, split, out_folder):
    '''
    For splitting the data across GPUs
    '''
    finished = [f.replace('.done', '') for f in os.listdir(out_folder) if f.endswith('.done')]
    
    files_to_keep = []
    with open('input_' + str(split), 'r') as infile:
        for line in infile:
            filename = line.strip()
            if filename in finished: continue # don't work on finished ones
            files_to_keep.append(filename)
    paths_to_keep = []
    for filename in result:
        short_name = filename.split('/')[-1].replace('.json.gz', '')
        if short_name in files_to_keep:
            paths_to_keep.append(filename)
    print("Keeping", len(paths_to_keep))
    return paths_to_keep

def parse_locs(in_path, out_folder, ent_dir, split, host, port):
    '''
    One file took 1.8 min on CPU
    '''
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    result = get_paths(result, split, out_folder)
    random.shuffle(result)

    os.makedirs(out_folder, exist_ok=True)

    geo = Geoparser(hosts=[host], port=port)

    for idx, filename in enumerate(result):
        start = time.time()
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        entities = defaultdict(list)
        with open(os.path.join(ent_dir, short_name + '.ents'), 'r') as infile:
            for line in infile:
                row = json.loads(line)
                url = row['url']
                for ent in row['ents']:
                    ent_type = ent[1]
                    ent_str = ent[0]
                    if ent_type == 'GPE' or ent_type == 'LOC':
                        entities[url].append(ent_str)

        ent_file = open(os.path.join(out_folder, short_name + '.jsonl'), 'w')
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                sent_ID_to_out = {}
                out = {}
                out['url'] = url
                if url not in entities:
                    ent_file.write(json.dumps(out) + '\n')
                else:
                    locs = entities[url].copy()
                    sents = text_to_sentences(text).split('\n')
                    for sent_idx, sent in enumerate(sents):
                        has_loc = False
                        for loc in locs:
                            if loc in sent:
                                has_loc = True
                                break
                        if has_loc:
                            out[sent_idx] = geo.geoparse_doc(sent)
                    ent_file.write(json.dumps(out) + '\n')
        ent_file.close()
        # for tracking finished shards
        log_file = open(os.path.join(out_folder, short_name + '.done'), 'w')
        log_file.write(str(time.time() - start) + '\n')
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--ent-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--split', required=True, type=int)
    parser.add_argument('--host', required=True, type=str)
    parser.add_argument('--port', required=True, type=int)

    args = parser.parse_args()
    parse_locs(args.data_dir, args.output_dir, args.ent_dir, args.split, args.host, args.port)
