"""
Run NER on about me pages: 
    save NER labels, tokens, and position
    
This uses the pipe environment
"""
import spacy
from tqdm import tqdm
import gzip
import os
from glob import glob
import json
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

def run_spacy(in_path, out_folder, split): 
    '''
    '''
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    result = get_paths(result, split)

    os.makedirs(out_folder, exist_ok=True)
    
    nlp = spacy.load('en_core_web_trf')
    spacy.require_gpu()
#     nlp.add_pipe('coreferee')
    
    for filename in tqdm(result): 
        url_ids = []
        texts = []
        short_name = filename.split("/")[-1].replace(".json.gz", "")
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                text = row["text"]
                url = row["id"]
                url_ids.append(url)
                texts.append(text[:1000000]) # account for spacy max len
        docs = nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        ent_file = open(os.path.join(out_folder, short_name + '.ents'), 'w')
#         coref_file = open(os.path.join(out_folder, short_name + '.coref'), 'w')
        for i, doc in enumerate(docs): 
            ent_d = {'url': url_ids[i], 'ents': []}
            for ent in doc.ents: 
                ent_d['ents'].append([ent.text, ent.label_])
            ent_file.write(json.dumps(ent_d) + '\n')
#             coref_d = {'url': url_ids[i], 'chains': []}
#             for chain in doc._.coref_chains:
#                 this_chain = []
#                 for mention in chain: 
#                     mention_d = {'ents':[]}
#                     mention_d['idx'] = mention.token_indexes
#                     span = doc[mention.token_indexes[0]:mention.token_indexes[-1] + 1]
#                     mention_d['text'] = span.text
#                     for ent in span.ents: 
#                         mention_d['ents'].append([ent.text, ent.label_])
#                     this_chain.append(mention_d)
#                 coref_d['chains'].append(this_chain)
#             coref_file.write(json.dumps(coref_d) + '\n')
        ent_file.close()
#         coref_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--split', required=True, type=int)

    args = parser.parse_args()
    run_spacy(args.data_dir, args.output_dir, args.split)