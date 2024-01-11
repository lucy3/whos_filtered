import argparse
import os
import numpy as np
import pandas as pd
import pickle
import torch
import uuid
from torch.utils.data import DataLoader
from train_clusterer import IterableDomain, get_files
from pathlib import Path
from tqdm.auto import tqdm
from kmeans_pytorch import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pdb
from glob import glob
import random
import json

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))

def example_in_cluster(text, vectorizer, kmeans, random_clusters=False):
    if random_clusters:       
        clusters = np.random.choice(range(kmeans.n_clusters), len(text))
    else:
        clusters = kmeans.predict(torch.from_numpy(vectorizer.transform(text)))
    return list(clusters)

def get_shortname(filename): 
    return filename.split('/')[-1].replace('.json.gz', '')

def cluster_file(filename, tfidf, kmeans, num_clusters, output_prefix):
    file_shortname = get_shortname(filename)
    Path(output_prefix).mkdir(parents=True, exist_ok=True)
    output = os.path.join(output_prefix, file_shortname)

    dataset = IterableDomain(files=[Path(filename)])
    dataloader = DataLoader(dataset, num_workers=0, batch_size=10000)
    zs = {}
    counter = 0
    for batch in dataloader:
        text = batch['text']
        ids = batch['id']
        cluster = example_in_cluster(text, tfidf, kmeans, random_clusters=False)
        for x, y in zip(ids, cluster): 
            zs[x] = y.item()
        counter += 1
    with open(output, 'w') as outfile: 
        json.dump(zs, outfile)
    with open(os.path.join(output_prefix, file_shortname + '_done.txt'), 'w') as outfile: 
        outfile.write('done\n')

def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--path-to-clusterer")
    parser.add_argument("--num-clusters")
    parser.add_argument("--output-prefix")

    cmd_args = parser.parse_args()

    kwargs = {}
    
    path_to_clusterer = Path(cmd_args.path_to_clusterer)
    kmeans = load_model(path_to_clusterer / "kmeans.pkl")
    tfidf = load_model(path_to_clusterer / "tfidf.pkl")

    # these are full file paths
    files = [y for x in os.walk(cmd_args.data_dir) for y in glob(os.path.join(x[0], '*.json.gz'))]
    print("torch.cuda.is_available:", torch.cuda.is_available()) # we don't want it to be available if local run
    
    files_done = []
    files_to_do = []
    for filename in files: 
        file_shortname = get_shortname(filename)
        if not os.path.exists(os.path.join(cmd_args.output_prefix, file_shortname + '_done.txt')): 
            files_to_do.append(filename)
        else: 
            files_done.append(filename)
        
    print("Num files to do:", len(files_to_do))
    print("Num files done:", len(files_done))

    for x in tqdm(files_to_do):
        cluster_file(x, tfidf, kmeans, cmd_args.num_clusters, cmd_args.output_prefix)