"""
Aggregates results across folds
and prints out best score for each model and set of params
"""
import os
import json
import numpy as np
from collections import defaultdict
from transformers import pipeline

home_path = '/home/lucyl/llm_social_identities/code/identity_measures/roberta_classifier'

def agg_results():
    model_results = defaultdict(list)
    for f in os.listdir(home_path):
        if f.startswith('.'): continue
        if os.path.isdir(f):
            results_path = os.path.join(home_path, f, 'word_test_results.json')
            if not os.path.exists(results_path): continue
            model = '-'.join(f.split('-')[:-1])
            with open(results_path, 'r') as infile:
                d = json.load(infile)
            model_results[model].append(d['R'])

    for model in model_results:
        precision = []
        recall = []
        f1 = []
        for res in model_results[model]:
            precision.append(res['precision'])
            recall.append(res['recall'])
            f1.append(res['f1'])
        print(model)
        print("Precision", round(np.mean(precision), 3), round(np.std(precision), 3))
        print("Recall", round(np.mean(recall), 3), round(np.std(recall), 3))
        print("F1", round(np.mean(f1), 3), round(np.std(f1), 3))
        print()

def show_predictions():
    '''
    Custom examples
    '''
    f = 'roberta_10ep-roles-2e-05'
    step = 225
    model_path = os.path.join(home_path, f, f'checkpoint-{step}')
    classifier = pipeline("ner", model=model_path)

    examples = ['I am a photographer and a chemist.', 'As a dancer, she is very talented.', 'My father is a banker.',
                'I love to listen to my favorite singer.', 'He saw a circus performer when he was twelve.']
    
    for res in classifier(examples): 
        for tok in res: 
            print(tok)
        print()

def examine_data():
    '''
    Manually look at validation data to see how many examples have pos example
    '''
    with open('reformat_annotated_examples.json', 'r') as infile:
        d = json.load(infile)

    curr_set_pos = 0
    for i in range(len(d)):
        if i % 100 == 0:
            print(i, curr_set_pos)
            curr_set_pos = 0
        if 'B-R' in set(d[str(i)]['entities']):
            curr_set_pos += 1

agg_results()
#show_predictions()
