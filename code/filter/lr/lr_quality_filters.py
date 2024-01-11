"""
This script trains text classifiers using code implemented by

Whose Language Counts as High Quality?
Measuring Language Ideologies in Text Data Selection
https://aclanthology.org/2022.emnlp-main.165
"""
from train import train_lr
from hyperparameters import BEST_HPS
import joblib
import os
import pandas as pd
import multiprocessing
from glob import glob
from tqdm import tqdm
import gzip
import json

def train_and_save_classifier(base_path, class_path): 
    '''
    This takes around ~5 min for the training step + a few min to load and dump stuff.
    '''
    train_path = os.path.join(base_path, 'train.jsonl')
    dev_path = os.path.join(base_path, 'dev.jsonl')
    test_path = os.path.join(base_path, 'test.jsonl')
    print("Loading train...")
    train = pd.read_json(train_path, lines=True)
    print("Loading dev...")
    dev = pd.read_json(dev_path, lines=True)
    print("Loading test...")
    test = pd.read_json(test_path, lines=True)
    print("Training...")
    clf, vectorizer, results = train_lr(train, dev, test, BEST_HPS)
    print(results)
    print("Dumping model and vectorizer...")
    joblib.dump(vectorizer, class_path + '_vectorizer.pkl')
    joblib.dump(clf, class_path + '_clf.pkl')
    
def process_batch(batch): 
    '''
    output: 
    {"id": page_url,"attributes": {metadata_name:[[0,3513,0.99]]}}
    '''
    short_name = batch["short_name"]
    out_folder = batch["out_folder"]
    filename = batch["filename"]
    vectorizer = batch["vectorizer"]
    clf = batch["clf"]
    attr_name = batch["attr_name"]
    
    out_file = open(os.path.join(out_folder, short_name + '.jsonl'), 'w')
    
    with gzip.open(filename, "rt") as infile:
        for line in infile:
            row = json.loads(line)
            text = row["text"]
            url = row["id"]
            score = clf.predict_proba(vectorizer.transform([text]))[0][1] # of positive class
            out_d = {'id': url, 'attributes': {attr_name: score}}
            out_file.write(json.dumps(out_d) + '\n')
    out_file.close()
    
def score_dataset(base_path, attr_name):  
    '''
    This takes around 8 min to run.
    '''
    class_path = os.path.join(base_path, attr_name)
    result = glob('/net/nfs/allennlp/lucyl/cc_data/cc_sample/**/*.json.gz', recursive=True)
    
    vectorizer = joblib.load(class_path + '_vectorizer.pkl')
    clf = joblib.load(class_path + '_clf.pkl')
    
    out_folder = "/home/lucyl/llm_social_identities/outputs/scores/" + attr_name
    os.makedirs(out_folder, exist_ok=True)
    
    batches = [
        {
            "filename": filename,
            "short_name": filename.split("/")[-1].replace(".json.gz", ""),
            "out_folder": out_folder,
            "vectorizer": vectorizer,
            "clf": clf,
            "attr_name": attr_name,
        }
        for filename in result
    ]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
        list(tqdm(p.imap(process_batch, batches), total=len(batches)))
    
if __name__ == '__main__':
    base_path = '/home/lucyl/llm_social_identities/data/filter_data/combined/WikiWebBooks'
#     train_and_save_classifier(base_path, os.path.join(base_path, 'wikiwebbooks'))
    score_dataset(base_path, 'wikiwebbooks')
    
    base_path = '/home/lucyl/llm_social_identities/data/filter_data/combined/WikiRef'
#     train_and_save_classifier(base_path, os.path.join(base_path, 'wikiref'))
    score_dataset(base_path, 'wikiref')
    
    base_path = '/home/lucyl/llm_social_identities/data/filter_data/combined/Wikipedia'
#     train_and_save_classifier(base_path, os.path.join(base_path, 'wikipedia'))
    score_dataset(base_path, 'wikipedia')
    
    base_path = '/home/lucyl/llm_social_identities/data/filter_data/combined/OpenWebText2'
#     train_and_save_classifier(base_path, os.path.join(base_path, 'openwebtext2'))
    score_dataset(base_path, 'openwebtext2')
    