'''
Conda environment: pipe
'''
import json
import spacy
from collections import defaultdict, Counter
import multiprocessing
from tqdm import tqdm
import os
from glob import glob
import csv
import gzip
import random
from sklearn.metrics import cohen_kappa_score, f1_score
import pandas as pd
from blingfire import text_to_sentences, text_to_words
import emoji

def sample_roles(base_folder, outfile_name, sample_size=600, frequency_cutoff=100, avoid_set=defaultdict(set)): 
    '''
    We want 500 examples for training, 100 examples for evaluation 
    '''
    in_folder = os.path.join(base_folder, "persona_occur")
    
    # reservoir sample one instance per occupation
    reservoir = defaultdict(tuple) # {term : (start, end, shorter filename, url)} 
    reservoir_count = Counter() # {term : count} 
    for f in tqdm(os.listdir(in_folder)): 
        shorter_name = f.split('/')[-1].replace('.jsonl', '')
        with open(os.path.join(in_folder, f), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['url']
                if url in avoid_set[shorter_name]: continue
                for r in d['personas']: 
                    spans = d['personas'][r]
                    for s in spans:
                        # take first example on page
                        start = s[0]
                        end = s[1]
                        break
                    if reservoir_count[r] == 0: 
                        reservoir[r] = (start, end, shorter_name, url)
                    else: 
                        j = random.randrange(reservoir_count[r] + 1)
                        if j == 0: # because sample size of n = 1
                            reservoir[r] = (start, end, shorter_name, url)
                    reservoir_count[r] += 1
                    
    random.seed(0)
    
    # sample sample_size terms that appear at least X times in dataset 
    freq_terms = [r for r in reservoir_count if reservoir_count[r] > frequency_cutoff]
    sample_terms = random.sample(freq_terms, sample_size)
    
    examples = defaultdict(dict)
    total = 0
    for r in reservoir_count: 
        if r not in sample_terms: continue
        start, end, shorter_name, url = reservoir[r]
        if url not in examples[shorter_name]: 
            examples[shorter_name][url] = []
        examples[shorter_name][url].append((start, end, r))
        total += 1
        
    assert total == sample_size
        
    with open(os.path.join(base_folder, outfile_name), 'w') as outfile: 
        json.dump(examples, outfile)

def get_person_list(): 
    '''
    Load the ngram-bucketed person list
    '''
    with open('ngram_buckets.json', 'r') as infile: # DATA + 'person_lists/ngram_buckets.json'
        d = json.load(infile)
    return d
        
def get_sample_sents(base_folder, infile_name, outfile_name): 
    buckets = get_person_list() 

    with open(os.path.join(base_folder, infile_name), 'r') as infile: 
        examples = json.load(infile)
    nlp = spacy.load('en_core_web_trf')
    in_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    result = glob(in_path + "/**/*.json.gz", recursive=True)
    res = {
        'shorter_name': [],
        'url': [],
        'r': [],
        'sentence': [],
        'start': [],
        'end': [],
    }
    for filename in tqdm(result): 
        shorter_name = filename.split('/')[-1].replace('.json.gz', '')
        if shorter_name not in examples: continue
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                row = json.loads(line)
                url = row['id']
                if url not in examples[shorter_name]: continue
                for tup in examples[shorter_name][url]: 
                    start, end, r = tup
                    text = row["text"]
                    doc = nlp.make_doc(text)
                    span = None
                    if r in buckets['1']: # character idx
                        assert r == text[start:end+1].lower()
                        case_r = text[start:end+1]
                        new_text = text[:start] + '#<' + case_r + '>#' + text[end+1:]
                        span = ('#<' + case_r + '>#', case_r, r)
                    else: # token idx
                        assert r == doc[start:end].text.lower()
                        case_r = doc[start:end].text
                        new_text = doc[:end-1].text + ' #<' + case_r.split()[-1] + '># ' + doc[end:].text
                        span = ('#<' + case_r.split()[-1] + '>#', case_r.split()[-1], r)
                    chunks = text_to_sentences(new_text).split('\n')
                    for chunk in chunks: 
                        if '#<' in chunk: 
                            new_new_text = chunk.replace(span[0], span[1])
                            start = chunk.index(span[0])
                            end = chunk.index(span[0]) + len(span[1])
                            assert span[1] == new_new_text[start:end]
                            res['shorter_name'].append(shorter_name)
                            res['url'].append(url)
                            res['r'].append(span[2])
                            res['sentence'].append(new_new_text)
                            res['start'].append(start)
                            res['end'].append(end)
    sample_size = 0
    for shorter_name in examples: 
        for url in examples[shorter_name]: 
            sample_size += len(examples[shorter_name][url])
    assert sample_size == len(res['r'])
    
    df = pd.DataFrame.from_dict(res)
    df.to_csv(os.path.join(base_folder, outfile_name))

def replace_emojis(text): 
    new_text = emoji.replace_emoji(text, replace="*")
    return new_text

def allocate_to_annotators(infile_name, input_folder, root, iaa_mapping): 
    '''
    There are seven authors, and everyone except lucy gets 80 annotations, lucy gets 120. 
    Each annotator also annotates 5 IAA examples.
    Binary classification of person identifies as the highlighted role, or they do not. 
    '''
    annotator_counts = {
                        'jesse': 80,
                        'suchin' : 80,
                        'lauren': 80,
                        'emma': 80, 
                        'luca': 80,
                        'david': 80, 
                        'lucy': 120, 
                        }
    
    annotator_fileids = {}
    num_iaa = 5
    for annotator in annotator_counts: 
        ids = list(range(annotator_counts[annotator] + num_iaa))
        random.shuffle(ids)
        annotator_fileids[annotator] = [annotator + '_' + str(i) for i in ids]
        
    orig_examples = {}
    with open(os.path.join(root, infile_name), 'r') as infile: 
        reader = csv.DictReader(infile) 
        for row in reader: 
            idx = int(row[''])
            sent = row['sentence']
            start = int(row['start'])
            end = int(row['end'])
            assert sent[start:end].lower() == row['r'].split()[-1]
            orig_examples[idx] = (sent, start, end)
            
    random_idx = list(orig_examples.keys())
    random.shuffle(random_idx)
    name2names = defaultdict(list)
    assignments = defaultdict(list)
    idx = 0
    for annotator in tqdm(annotator_counts): 
        for i in range(annotator_counts[annotator]): 
            sent, start, end = orig_examples[idx]
            f = annotator_fileids[annotator][i]

            text_file = open(os.path.join(input_folder, f + '.txt'), 'w')
            text_file.write(replace_emojis(sent.strip()))
            text_file.close()

            ann_file = open(os.path.join(input_folder, f + '.ann'), 'w')
            ann_file.write('T1' + '\t' + 'Role_Identify ' + str(start) + ' ' + \
                          str(end) + '\t' + sent[start:end] + '\n')
            ann_file.close()
            name2names[idx].append(f)
            assignments[annotator].append(idx)
            idx += 1
                
    already_double = set()
    for annotator in tqdm(annotator_counts): 
        total = 0
        while total < num_iaa: 
            samp = random.sample(random_idx, 1) 
            if samp[0] not in assignments[annotator] and samp[0] not in already_double: 
                idx = samp[0]
                already_double.add(idx)
                sent, start, end = orig_examples[idx]
                f = annotator_fileids[annotator][annotator_counts[annotator] + total]
                text_file = open(os.path.join(input_folder, f + '.txt'), 'w')
                text_file.write(replace_emojis(sent.strip()))
                text_file.close()

                ann_file = open(os.path.join(input_folder, f + '.ann'), 'w')
                ann_file.write('T1' + '\t' + 'Role_Identify ' + str(start) + ' ' + \
                              str(end) + '\t' + sent[start:end] + '\n')
                ann_file.close()
                name2names[idx].append(f)
                total += 1
                
    doubly_count = 0
    for name in name2names: 
        assert len(name2names[name]) < 3
        if len(name2names[name]) == 2: 
            doubly_count += 1
    assert doubly_count == len(annotator_counts)*num_iaa
    assert len(name2names) == len(orig_examples)
        
    expected_total = sum(annotator_counts.values()) + num_iaa*len(annotator_counts)
    assert expected_total == len(os.listdir(input_folder)) / 2
                
    with open(os.path.join(root, iaa_mapping), 'w') as outfile: 
        json.dump(name2names, outfile)
        
def get_annotations(full_ann_path): 
    span_to_identify = {}
    span_to_text = {}
    with open(full_ann_path, 'r') as infile:
        for line in infile: 
            contents = line.strip().split('\t')
            parts = contents[1].split()
            if line.startswith('T'): 
                t_id = contents[0]
                relation = parts[0]
                assert relation == 'Role_Identify'
                start = parts[1]
                end = parts[2] 
                text = contents[2]
                span_to_identify[(start, end)] = relation
                span_to_text[(start, end)] = text
    return span_to_identify, span_to_text
        
def interannotator_agreement(ann_path, iaa_mapping, root): 
    '''
    A third annotator (me, unless the initial disagremeent involves me) resolves disagreements
    '''
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)
    disagreements_file = open(os.path.join(root, 'roles_disagree.csv'), 'w')
    writer = csv.writer(disagreements_file)
    identify1 = []
    identify2 = []
    disagree_count = 0
    for orig_name in d: 
        annotated = d[orig_name]
        if len(annotated) == 1: continue
        assert len(annotated) == 2
        annotated1 = annotated[0]
        annotated2 = annotated[1]
        span_to_identify1, span_to_text1 = get_annotations(os.path.join(ann_path, annotated1 + '.ann')) 
        span_to_identify2, span_to_text2 = get_annotations(os.path.join(ann_path, annotated2 + '.ann')) 
        spans = set(span_to_identify1.keys()) | set(span_to_identify2.keys())
        for span in spans: 
            if span in span_to_text1: 
                text = span_to_text1[span]
            else: 
                text = span_to_text2[span]
            identify1_bool = 0
            identify2_bool = 0
            if span in span_to_identify1: 
                identify1_bool = 1
            if span in span_to_identify2: 
                identify2_bool = 1
            # print out disagreements 
            if identify1_bool != identify2_bool: 
                print("Identify disagree:", span, annotated1, annotated2, text)
                writer.writerow(['identify_role', span, annotated1, annotated2, text])
        ann1_keys = '-'.join([str(tup[0]) + '_' + str(tup[1]) for tup in sorted(span_to_identify1.keys())])
        ann2_keys = '-'.join([str(tup[0]) + '_' + str(tup[1]) for tup in sorted(span_to_identify2.keys())])
        if ann1_keys != ann2_keys: disagree_count += 1
        identify1.append(ann1_keys)
        identify2.append(ann2_keys)
    print(identify1)
    print(identify2)
    disagreements_file.close()
    print("Disagree count:", disagree_count, "--- Total number of iaa examples:", len(identify1))
    print("Cohen kappa agreement for identify:", cohen_kappa_score(identify1, identify2))
#     print("F1 agreement for identify:", f1_score(identify1, identify2))
    
def reformat_annotations(annotated, ann_path, tiebreakers):  
    ann = annotated[0]
    if len(annotated) == 1: 
        full_ann_path = os.path.join(ann_path, ann + '.ann')
        span_to_text = {}
        with open(full_ann_path, 'r') as infile:
            for line in infile: 
                contents = line.strip().split('\t')
                parts = contents[1].split()
                if line.startswith('T'): 
                    t_id = contents[0]
                    relation = parts[0]
                    assert relation == 'Role_Identify'
                    start = parts[1]
                    end = parts[2] 
                    text = contents[2]
                    span_to_text[int(start)] = (text, int(end))
    elif len(annotated) == 2: 
        span_to_text = {}
        for ann in annotated: 
            full_ann_path = os.path.join(ann_path, ann + '.ann')
            with open(full_ann_path, 'r') as infile:
                for line in infile: 
                    contents = line.strip().split('\t')
                    parts = contents[1].split()
                    if line.startswith('T'): 
                        t_id = contents[0]
                        relation = parts[0]
                        assert relation == 'Role_Identify'
                        start = parts[1]
                        end = parts[2] 
                        span = (start, end)
                        text = contents[2]
                        if (annotated[0], annotated[1], span) in tiebreakers: 
                            print("excluding", (annotated[0], annotated[1], span), text)
                            continue
                        span_to_text[int(start)] = (text, int(end))
                        
                        
    sorted_keys = sorted(span_to_text.keys())
    full_text_path = os.path.join(ann_path, ann + '.txt')
    tokens = []
    entities = []
    with open(full_text_path, 'r') as infile: 
        example = infile.read()
        before = 0
        for start in sorted_keys: 
            text, end = span_to_text[start]
            assert example[start:end] == text
            these_tokens = text_to_words(example[before:start]).split(' ')
            tokens.extend(these_tokens)
            entities.extend(['O' for t in these_tokens])
            tokens.append(text)
            entities.append('B-R')
            before = end
        these_tokens = text_to_words(example[before:]).split(' ')
        tokens.extend(these_tokens)
        entities.extend(['O' for t in these_tokens])
    assert len(tokens) == len(entities)
    return tokens, entities
    
def load_tie_breakers(input_file): 
    # return (annotator1, annotator2, span) : winner}
    ret = set()
    with open(input_file, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            if row['action'] == 'exclude': 
                span = tuple(row['indices'].replace('(', '').replace(')', '').replace('\'', '').split(', '))
                ret.add((row['annotator1'], row['annotator2'], span))
    return ret

def save_tokens_with_entities(iaa_mapping, root, ann_paths):
    '''
    Saves a json of orig_name to a zipped list of blingfire tokens, e.g. 
    {'tokens': ['I', 'am', 'a', 'chemist', '.'], 'entities': ['O', 'O', 'O', 'R', 'O']}
    
    ann_paths[0] is the original annotated dataset
    ann_paths[1] is the additional annotated data
    '''
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)
        
    tiebreakers = load_tie_breakers(os.path.join(root, 'roles_disagree_resolved.csv'))
    
    # from the annotation path, load up examples 
    output = defaultdict(dict)
    for orig_name in d: 
        annotated = d[orig_name]
        tokens, entities = reformat_annotations(annotated, ann_paths[0], tiebreakers) 
        output[orig_name]['tokens'] = tokens
        output[orig_name]['entities'] = entities
        
    for f in os.listdir(ann_paths[1]): 
        if not f.endswith('.txt'): continue
        ann_name = f.replace('.txt', '')
        tokens, entities = reformat_annotations([ann_name], ann_paths[1], set()) 
        output[ann_name]['tokens'] = tokens
        output[ann_name]['entities'] = entities
        
    print("Total number of examples:",len(output))
        
    with open('./roberta_classifier/reformat_annotated_examples.json', 'w') as outfile: 
        json.dump(output, outfile)
        
def annotation_round1(): 
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    original_path = root + 'roles_annotate_assign/'
    ann_path = root + 'roles_annotate_done/'
    iaa_mapping = root + 'roles_eval_name2names.json'    
    #sample_roles(root, 'persona_examples_to_annotate.json')
    #get_sample_sents(root, 'persona_examples_to_annotate.json', 'persona_examples_to_annotate.csv')
    #allocate_to_annotators('persona_examples_to_annotate.csv', original_path, root, iaa_mapping)
    #interannotator_agreement(ann_path, iaa_mapping, root)
    
def reformat_to_brat(infile_name, input_folder, root):
    os.makedirs(input_folder, exist_ok=True) 
    
    orig_examples = {}
    with open(os.path.join(root, infile_name), 'r') as infile: 
        reader = csv.DictReader(infile) 
        for row in reader: 
            idx = int(row[''])
            sent = row['sentence']
            start = int(row['start'])
            end = int(row['end'])
            assert sent[start:end].lower() == row['r'].split()[-1]
            orig_examples[idx] = (sent, start, end)
            
    random_idx = list(orig_examples.keys())
    random.shuffle(random_idx)
    for idx in orig_examples: 
        sent, start, end = orig_examples[idx]
        f = 'additional_' + str(idx)

        text_file = open(os.path.join(input_folder, f + '.txt'), 'w')
        text_file.write(replace_emojis(sent.strip()))
        text_file.close()

        ann_file = open(os.path.join(input_folder, f + '.ann'), 'w')
        ann_file.write('T1' + '\t' + 'Role_Identify ' + str(start) + ' ' + \
                      str(end) + '\t' + sent[start:end] + '\n')
        ann_file.close()
        
def get_avoid_set(base_folder, infile_name): 
    with open(os.path.join(base_folder, infile_name), 'r') as infile: 
        examples = json.load(infile)
        
    urls = defaultdict(set)
    for shorter_name in examples: 
        urls[shorter_name] = set(examples[shorter_name].keys())
    return urls
    
def annotation_additions(): 
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    original_path = root + 'roles_add_annotate_assign/'
    ann_path = root + 'roles_add_annotate_done/'
#     avoid_set = get_avoid_set(root, 'persona_examples_to_annotate.json')
#     sample_roles(root, 'roles_add_examples_to_annotate.json', sample_size=400, avoid_set=avoid_set)
#     get_sample_sents(root, 'roles_add_examples_to_annotate.json', 'roles_add_examples_to_annotate.csv')
#     reformat_to_brat('roles_add_examples_to_annotate.csv', original_path, root)

def ann_outcome_stats(): 
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    ann_path1 = root + 'roles_annotate_done/'
    ann_path2 = root + 'roles_add_annotate_done/'
    iaa_mapping = root + 'roles_eval_name2names.json' 
    
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)
    yes = 0 # has roles in doc
    total_roles = 0
    total_files = 0
    
    # round 1
    for orig_name in d: 
        total_files += 1
        annotated = d[orig_name]
        has_ann = False
        spans = set()
        for ann in annotated: 
            span_to_identify1, span_to_text1 = get_annotations(os.path.join(ann_path1, ann + '.ann')) 
            num_ann = len(span_to_identify1)
            if num_ann > 0: 
                has_ann = True
            spans.update(list(span_to_identify1.keys()))
        total_roles += len(spans)
        if has_ann: 
            yes += 1
    # round 2
    for f in os.listdir(ann_path2): 
        if not f.endswith('.ann'): continue
        total_files += 1
        span_to_identify1, span_to_text1 = get_annotations(os.path.join(ann_path2, f)) 
        num_ann = len(span_to_identify1)
        if num_ann > 0: 
            yes += 1
        total_roles += num_ann
            
    print(yes, total_files, total_roles)
    
def save_ann_data_main(): 
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    ann_paths = [root + 'roles_annotate_done/', root + 'roles_add_annotate_done/']
    iaa_mapping = root + 'roles_eval_name2names.json'    
    
    save_tokens_with_entities(iaa_mapping, root, ann_paths)

if __name__ == "__main__":
    ann_outcome_stats()
    