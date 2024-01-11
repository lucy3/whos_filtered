"""
Copied from: https://raw.githubusercontent.com/kernelmachine/quality-filter/main/lr/train.py
"""

import argparse
import json
import logging
import os
import pathlib
import random
import shutil
import sys
import time
from ast import literal_eval
from shutil import rmtree
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import ray
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer, HashingVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from hyperparameters import (SEARCH_SPACE,BEST_HPS, HyperparameterSearch,
                             RandomSearch)
from util import jackknife, replace_bool, stratified_sample

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_lr(train,
             dev,
             test,
             search_space):
    master = pd.concat([train, dev], 0)
    space = HyperparameterSearch(**search_space)
    sample = space.sample()
    if sample.pop('stopwords') == 1:
        stop_words = 'english'
    else:
        stop_words = None
    weight = sample.pop('weight')
    if weight == 'binary':
        binary = True
    else:
        binary = False
    ngram_range = sample.pop('ngram_range')
    ngram_range = tuple(sorted([int(x) for x in ngram_range.split()]))
    if weight == 'tf-idf':
        vect = TfidfVectorizer(stop_words=stop_words,
                               lowercase=True,
                               ngram_range=ngram_range)
    elif weight == 'hash':
        vect = HashingVectorizer(stop_words=stop_words, lowercase=True, ngram_range=ngram_range)
    else:
        vect = CountVectorizer(binary=binary,
                               stop_words=stop_words,
                               lowercase=True,
                               ngram_range=ngram_range)
    start = time.time()
    vect.fit(tqdm(master.text, desc="fitting data", leave=False))
    X_train = vect.transform(tqdm(train.text, desc="transforming training data",  leave=False))
    X_dev = vect.transform(tqdm(dev.text, desc="transforming dev data",  leave=False))
    if test is not None:
        X_test = vect.transform(tqdm(test.text, desc="transforming test data",  leave=False))

    sample['C'] = float(sample['C'])
    sample['tol'] = float(sample['tol'])
    classifier = LogisticRegression(**sample)
    classifier.fit(X_train, train.label)
    end = time.time()
    for k, v in sample.items():
        if not v:
            v = str(v)
        sample[k] = [v]
    res = pd.DataFrame(sample)
    preds = classifier.predict(X_dev)
    if test is not None:
        test_preds = classifier.predict(X_test)
    res['dev_f1'] = f1_score(dev.label, preds, average='macro')
    if test is not None:
        res['test_f1'] = f1_score(test.label, test_preds, average='macro')
    res['dev_accuracy'] = classifier.score(X_dev, dev.label)
    if test is not None:
        res['test_accuracy'] = classifier.score(X_test, test.label)
    res['training_duration'] = end - start
    res['ngram_range'] = str(ngram_range)
    res['weight'] = weight
    res['stopwords'] = stop_words
    return classifier, vect, res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str, required=False)
    parser.add_argument('--test_file', type=str, required=False)
    parser.add_argument('--search_trials', type=int, default=5)
    parser.add_argument('--train_subsample', type=int, required=False)
    parser.add_argument('--stratified', action='store_true')
    parser.add_argument('--jackknife_partitions', type=int, default=5, required=False)
    parser.add_argument('--save_jackknife_partitions', action='store_true')
    parser.add_argument('--serialization_dir', '-s', type=str)
    parser.add_argument('--override', '-o', action='store_true')
    parser.add_argument('--evaluate_on_test', '-t', action='store_true')


    args = parser.parse_args()

    if not os.path.isdir(args.serialization_dir):
        os.makedirs(args.serialization_dir)
    else:
        if args.override:
            rmtree(args.serialization_dir)
            os.makedirs(args.serialization_dir)
        else:
            print(f"serialization directory {args.serialization_dir} exists. Aborting! ")
    print(f"reading training data at {args.train_file}...")
    train = pd.read_json(args.train_file, lines=True)
    if args.train_subsample:
        if args.stratified:
            train = stratified_sample(train, "label", args.train_subsample)
        else:
            train = train.sample(n=args.train_subsample)

    if args.dev_file:
        print(f"reading dev data at {args.dev_file}...")
        dev = pd.read_json(args.dev_file, lines=True)
    else:
        print("Dev file not provided, will jackknife training data...")

    if args.evaluate_on_test:
        if args.test_file:
            print(f"reading test data at {args.test_file}...")
            test = pd.read_json(args.test_file, lines=True)
        else:
            print("Test file not provided.")
            sys.exit(1)
    else:
        test = None

    num_assignments = args.search_trials
    num_partitions = args.jackknife_partitions
    df = pd.DataFrame()
    current_f1 = 0.0
    best_classifier = None
    best_vect = None
    if args.dev_file:
        pbar = tqdm(range(num_assignments), desc="search trials", leave=False)
        for i in pbar:
            try:
                classifier, vect, res = train_lr(train, dev, test, BEST_HPS)
                df = pd.concat([df, res], 0, sort=True)
                best_f1 = df.dev_f1.max()
                if res.dev_f1[0] > current_f1:
                    current_f1 = res.dev_f1[0]
                    best_classifier = classifier
                    best_vect = vect
                pbar.set_description(f"mean +- std dev F1: {df.dev_f1.mean()} +- {df.dev_f1.std()}")
            except KeyboardInterrupt:
                break
    else:
        if args.save_jackknife_partitions:
            if not os.path.isdir(os.path.join(args.serialization_dir, "jackknife")):
                os.mkdir(os.path.join(args.serialization_dir, "jackknife"))
        for ix, (train, dev) in tqdm(enumerate(jackknife(train, num_partitions=num_partitions)),
                                     total=num_partitions,
                                     leave=False,
                                     desc="jackknife partitions"):
            for i in tqdm(range(num_assignments), desc="search trials",  leave=False):
                classifier, vect, res = train_lr(train, dev, test, SEARCH_SPACE)
                df = pd.concat([df, res], 0, sort=True)
                best_f1 = df.dev_f1.max()
                if res.dev_f1[0] > current_f1:
                    current_f1 = res.dev_f1[0]
                    best_classifier = classifier
                    best_vect = vect
            df['dataset_reader.sample'] = train.shape[0]
            df['model.encoder.architecture.type'] = 'logistic regression'
            if args.save_jackknife_partitions:
                train.to_json(
                    os.path.join(args.serialization_dir,
                                 "jackknife",
                                 f"train.{ix}"),
                                 lines=True,
                                 orient="records")
                dev.to_json(os.path.join(args.serialization_dir,
                                         "jackknife",
                                         f"dev.{ix}"),
                                         lines=True,
                                         orient='records')

    print("DEV STATISTICS")
    print("================")
    print(f"mean +- std F1: {df.dev_f1.mean()} +- {df.dev_f1.std()}")
    print(f"max F1: {df.dev_f1.max()}")
    print(f"min F1: {df.dev_f1.min()}")
    print(f"mean +- std accuracy: {df.dev_accuracy.mean()} +- {df.dev_accuracy.std()}")
    print(f"max accuracy: {df.dev_accuracy.max()}")
    print(f"min accuracy: {df.dev_accuracy.min()}")
    print("")
    print("BEST HYPERPARAMETERS")
    print(f"=====================")
    best_hp = df.reset_index().iloc[df.reset_index().dev_f1.idxmax()].to_dict()
    print(df.reset_index().iloc[df.reset_index().dev_f1.idxmax()])

    if test is not None:
        print("TEST STATISTICS")
        print("================")
        print(f"mean +- std F1: {df.test_f1.mean()} +- {df.test_f1.std()}")
        print(f"max F1: {df.test_f1.max()}")
        print(f"min F1: {df.test_f1.min()}")
        print(f"mean +- std accuracy: {df.test_accuracy.mean()} +- {df.test_accuracy.std()}")
        print(f"max accuracy: {df.test_accuracy.max()}")
        print(f"min accuracy: {df.test_accuracy.min()}")

    df.to_json(os.path.join(args.serialization_dir, "results.jsonl"), lines=True, orient='records')
    with open(os.path.join(args.serialization_dir, "best_hyperparameters.json"), "w+") as f:
        best_hp = df.reset_index().iloc[df.reset_index().dev_f1.idxmax()].to_dict()
        for k,v in best_hp.items():
            if isinstance(v, np.int64):
                best_hp[k] = int(v)
            if isinstance(v, str) and "[" in v:
                v = literal_eval(v)
                best_hp[k] = f"{v[0]} {v[1]}"
        best_hp.pop("index")
        best_hp.pop("dev_accuracy")
        best_hp.pop("dev_f1")
        if test is not None:
            best_hp.pop("test_accuracy")
            best_hp.pop("test_f1")
        best_hp.pop("training_duration")
        json.dump(best_hp, f)
    if best_hp['weight'] != "hash":
        with open(os.path.join(args.serialization_dir, "vocab.json"), 'w+') as f:
            for k,v in best_vect.__dict__['vocabulary_'].items():
                best_vect.__dict__['vocabulary_'][k] = int(v)
            json.dump(best_vect.__dict__['vocabulary_'], f)

    os.mkdir(os.path.join(args.serialization_dir, "archive"))
    try:
        np.save(os.path.join(args.serialization_dir, "archive", "idf.npy"), best_vect.idf_)
    except:
        pass
    np.save(os.path.join(args.serialization_dir, "archive", "classes.npy"),best_classifier.classes_)
    np.save(os.path.join(args.serialization_dir, "archive", "coef.npy"),best_classifier.coef_)
    np.save(os.path.join(args.serialization_dir, "archive", "intercept.npy"), best_classifier.intercept_)