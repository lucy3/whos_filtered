"""
Sample 200 about pages for correcting / checking geoparses
"""
import os
import random
from collections import defaultdict, Counter
import json
from blingfire import text_to_sentences
from tqdm import tqdm
import gzip
import emoji

def sample_predictions(): 
    '''
    Since the first chunk of predicted data is just tail pages
    due to lack of input shuffling, we take one random
    about page from the 200 most recently finished splits. 
    
    We track which split each page is from so that we
    can easily find the original bios in the next step. 
    '''
    return # we only run this once 
    
    num_samples = 200
    geoparse_folder = '/home/lucyl/llm_social_identities/outputs/identity/geoparse'
    files = [f for f in os.listdir(geoparse_folder) if f.endswith('.done')]
    x = []
    y = []
    for f in files: 
        mtime = os.path.getmtime(os.path.join(geoparse_folder, f))
        x.append(mtime)
        y.append(f.replace('.done', ''))
    x, y = zip(*sorted(zip(x, y)))
    y = y[-num_samples:] # output of geoparsing was in random order already
    samples = defaultdict(dict) # {short_name: {'pred': bio predictions, 'text': text}}
    for f in tqdm(y): 
        random.seed(0)
        num_seen = 0
        with open(os.path.join(geoparse_folder, f + '.jsonl'), 'r') as infile: 
            for line in infile: 
                num_seen += 1
        
        samp = random.randint(0, num_seen)
                
        num_seen = 0
        with open(os.path.join(geoparse_folder, f + '.jsonl'), 'r') as infile: 
            for line in infile: 
                if num_seen == samp: 
                    samples[f]['pred'] = json.loads(line)
                    break
                num_seen += 1
                
    bio_folder = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    
    for f in tqdm(samples): 
        url = samples[f]['pred']['url']
        if 'middle' in f: 
            mid_folder = 'cc_en_middle'
        if 'tail' in f: 
            mid_folder = 'cc_en_tail'
        if 'head' in f: 
            mid_folder = 'cc_en_head'
        with gzip.open(os.path.join(bio_folder, mid_folder, f + '.json.gz'), "rt") as infile:
            for line in infile: 
                row = json.loads(line)
                row_url = row["id"]
                if row_url == url: 
                    text = row['text']
                    samples[f]['text'] = text
                    break
    
    with open('/home/lucyl/llm_social_identities/outputs/identity/geo_annotate.json', 'w') as outfile: 
        json.dump(samples, outfile)
        
def replace_emojis(text): 
    new_text = emoji.replace_emoji(text, replace="*")
    return new_text

def format_for_brat(): 
    '''
    The bio would be in the .txt file, while the predictions would be in the .ann file. 
    
    Example .ann file: 
    T1	Associated 14 19	Wales
    #1	AnnotatorNotes T1	https://www.geonames.org/2634895/
    T2	Associated 679 695	Golden Gate Park
    #2	AnnotatorNotes T2	https://www.geonames.org/5352860/
    
    After running this, see if the characters align properly (in
    an earlier project there was the issue where emojis would mess with
    character indices)
    '''
    with open('/home/lucyl/llm_social_identities/outputs/identity/geo_annotate.json', 'r') as infile: 
        samples = json.load(infile)
        
    orig_input_folder = '/home/lucyl/llm_social_identities/outputs/identity/geo_annotate_original'
        
    for f in samples: 
        text = samples[f]['text']
        sents = text_to_sentences(text).split('\n')
        preds = samples[f]['pred']
        curr_char_idx = 0
        ent_num = 0
        
        text_file = open(os.path.join(orig_input_folder, f + '.txt'), 'w')
        ann_file = open(os.path.join(orig_input_folder, f + '.ann'), 'w')
        
        for i, sent in enumerate(sents): 
            text_file.write(replace_emojis(sent.strip()) + '\n')
            if str(i) in preds: 
                assert sent == preds[str(i)]['doc_text']
                for ent in preds[str(i)]['geolocated_ents']: 
                    ent_num += 1
                    text_span = ent['search_name']
                    start = ent["start_char"]
                    end = start + len(text_span)
                    if 'geonameid' in ent: 
                        span_geonameid = ent['geonameid']
                    else: 
                        span_geonameid = 'None'
                    ann_file.write('T' + str(ent_num) + '\t' + 'Associated ' + str(curr_char_idx + start) + ' ' + \
                          str(curr_char_idx + end) + '\t' + text_span + '\n')
                    ann_file.write('#' + str(ent_num) + '\tAnnotatorNotes T' + str(ent_num) + \
                          '\thttps://www.geonames.org/' + str(span_geonameid) + '/'  + '\n')
            curr_char_idx += len(sent) + 1
        text_file.close()
        ann_file.close()

def allocate_to_annotators(): 
    '''
    This file makes a new folder, renames files, and duplicates
    ones we are using to calculate IAA. 
    
    files are named firstname_#
    '''
#     annotator_counts = {
#                         'jesse': 50,
#                         'emma': 50, 
#                         'luca': 50,
#                         'lucy': 50, 
#                         }
    annotator_counts = {
                        'jesse': 25,
                        'suchin' : 25,
                        'lauren': 25,
                        'emma': 25, 
                        'luca': 25,
                        'david': 25, 
                        'lucy': 50, 
                        }
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    orig_input_folder = root + 'geo_annotate_original'
    new_input_folder = root + 'geo_annotate_assign'
    assert sum(annotator_counts.values()) == (len(os.listdir(orig_input_folder)) / 2)
    
    random.seed(0)
    orig_filenames = [f.replace('.txt', '') for f in os.listdir(orig_input_folder) if f.endswith('.txt')]
    random.shuffle(orig_filenames)
    assignments = defaultdict(list)
    idx = 0
    for annotator in annotator_counts: 
        for i in range(annotator_counts[annotator]): 
            assignments[annotator].append(orig_filenames[idx])
            idx += 1
    
    all_filenames = set()
    for annotator in assignments: 
        assert len(assignments[annotator]) == annotator_counts[annotator]
        all_filenames.update(assignments[annotator])
    assert len(all_filenames) == len(orig_filenames)
    
    # some examples will be doubly annotated for computing IAA
#     num_iaa = 10
    num_iaa = 5
    already_double = set()
    for annotator in assignments: 
        total = 0
        while total < num_iaa: 
            samp = random.sample(orig_filenames, 1) 
            if samp[0] not in assignments[annotator] and samp[0] not in already_double: 
                assignments[annotator].append(samp[0])
                already_double.add(samp[0])
                total += 1
                
    # a mapping from original filename to list of new filenames
    name2names = defaultdict(list)
    for annotator in assignments: 
        file_num = 0
        for f in assignments[annotator]: 
            new_f = annotator + '_' + str(file_num)
            os.system('cp ' + os.path.join(orig_input_folder, f + '.txt') + ' ' + os.path.join(new_input_folder, new_f + '.txt'))
            os.system('cp ' + os.path.join(orig_input_folder, f + '.ann') + ' ' + os.path.join(new_input_folder, new_f + '.ann'))
            file_num += 1
            name2names[f].append(new_f)
            
    for name in name2names: 
        assert len(name2names[name]) < 3
        
    with open(root + 'geo_eval_name2names.json', 'w') as outfile: 
        json.dump(name2names, outfile)
        
if __name__ == "__main__":
    #sample_predictions()
    #format_for_brat()
    #allocate_to_annotators()
    pass
