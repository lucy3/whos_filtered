from glob import glob
import zreader
import ujson as json
from tqdm import tqdm
import random

in_path = '/shared/kitaev/pile/train/*.jsonl.zst'
num_tokens=300000000
files = glob(in_path, recursive=True)
id_to_len = {}
idx = 0
for f in tqdm(sorted(files)):
    if not f.endswith('00.jsonl.zst'): continue # one shard is enough
    reader = zreader.Zreader(f, chunk_size=8192)
    for line in tqdm(reader.readlines()):
        d = json.loads(line)
        if d['meta']['pile_set_name'] != 'OpenWebText2': continue
        id_to_len[idx] = len(d['text'].split())
        idx += 1

sample = list(id_to_len.keys())
random.shuffle(sample)
total = 0
idx_to_keep = set()
for idx in sample:
    if total + id_to_len[idx] > num_tokens: 
        print("Goal number of tokens met!")
        break
    total += id_to_len[idx]
    idx_to_keep.add(idx)

with open('all.jsonl', 'w') as outfile: 
    idx = 0
    for f in tqdm(sorted(files)): 
        if not f.endswith('00.jsonl.zst'): continue 
        reader = zreader.Zreader(f, chunk_size=8192)
        for line in reader.readlines(): 
            d = json.loads(line)     
            if d['meta']['pile_set_name'] != 'OpenWebText2': continue  
            if idx in idx_to_keep: 
                out = {}
                out['id'] = d['meta']['pile_set_name'] + '_' + str(idx)
                out['text'] = d['text']
                out['label'] = 1
                outfile.write(json.dumps(out) + '\n')
            idx += 1
