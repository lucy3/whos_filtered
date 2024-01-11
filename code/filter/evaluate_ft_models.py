from fasttext.FastText import _FastText
import os
import json
from tqdm import tqdm

in_path = '/home/lucyl/llm_social_identities/data/filter_data/combined/WikiWebBooks/'
model_path = '/home/lucyl/llm_social_identities/data/filter_data/wikiwebbooks_cc.bin'
model = _FastText(model_path)

dev_file = os.path.join(in_path, 'dev.jsonl')
total = 0
correct = 0
with open(dev_file, 'r') as infile: 
    for line in tqdm(infile): 
        total += 1
        d = json.loads(line)
        pred = model.predict(d['text'].replace('\n', ' '))[0][0]
        if pred == '__label__wikiwebbooks' and d['label'] == 1: 
            correct += 1
        elif pred == '__label__random_cc' and d['label'] == 0: 
            correct += 1
print("Dev accuracy:", correct / total)