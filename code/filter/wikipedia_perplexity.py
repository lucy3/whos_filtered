"""
Calculate perplexity on a sample of Wikipedia
"""
import kenlm
import os
import text_normalizer
from tqdm import tqdm
import json
import sentencepiece

def pp(log_score, length):
    return 10.0 ** (-log_score / length)

model = kenlm.Model('/home/lucyl/cc_net/data/lm_sp/en.arpa.bin')
sp = sentencepiece.SentencePieceProcessor()
sp.load('/home/lucyl/cc_net/data/lm_sp/en.sp.model')

wikipedia_folder = '/home/lucyl/llm_social_identities/data/filter_data/split/Wikipedia'
wikipedia_path = os.path.join(wikipedia_folder, 'all.jsonl')

results = {}
with open(wikipedia_path, 'r') as infile: 
    for line in tqdm(infile): 
        d = json.loads(line)
        text = d['text']
        text = text_normalizer.normalize(text)
        idx = d['id']
        tokenized = sp.encode_as_pieces(text)
        text = " ".join(tokenized)
        lines = text.split('\n')
        doc_log_score, doc_length = 0, 0
        for line in lines:
            line = text_normalizer.normalize(line)
            log_score = model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length

        perplexity = round(pp(doc_log_score, doc_length), 1)
        results[idx] = perplexity
with open(os.path.join(wikipedia_folder, 'wiki_perplexity.json'), 'w') as outfile: 
    json.dump(results, outfile)