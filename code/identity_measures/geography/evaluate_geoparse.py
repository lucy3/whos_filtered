"""
Compare the human-edited labels
with the automatically generated ones.
"""
import os
import json
from collections import defaultdict, Counter
from mordecai3.elastic_utilities import make_conn, get_entry_by_id
from sklearn.metrics import cohen_kappa_score
import csv

def sanity_check(original_path, ann_path):
    # see which docs are exactly same as original
    not_changed_files = []
    for f in os.listdir(original_path):
        if not f.endswith('.ann'): continue
        assert os.path.exists(os.path.join(ann_path, f))
        ann_file = os.path.join(ann_path, f)
        ann_lines = []
        with open(ann_file, 'r') as infile:
            only_assoc = True
            for line in infile:
                ann_lines.append(line.strip())
                if 'Identify' in line: 
                    only_assoc = False
        if only_assoc and len(ann_lines) > 0: 
            print("NO CHANGE IN IDENTIFY LABELS:", ann_file)
        orig_file = os.path.join(original_path, f)
        changed = False
        i = 0
        with open(orig_file, 'r') as infile:
            for i, line in enumerate(infile):
                if len(ann_lines) == 0 or line.strip() != ann_lines[i]:
                    changed = True
                    break
        if i != 0 and not changed and len(ann_lines) > 0:
            not_changed_files.append(f)
    for f in sorted(not_changed_files):
        print("NO CHANGE:", f)
    print('\n-----------------\n')
    
def sanity_check_doubly_annotated(original_path, iaa_mapping): 
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)
    for orig_name in d: 
        annotated = d[orig_name]
        if len(annotated) == 1: continue
        assert len(annotated) == 2
        annotated1 = annotated[0]
        annotated2 = annotated[1] 
        with open(os.path.join(original_path, annotated1 + '.txt'), 'r') as infile: 
            one = infile.read()
        with open(os.path.join(original_path, annotated2 + '.txt'), 'r') as infile: 
            two = infile.read()
        assert one == two
        
def get_annotations(full_ann_path): 
    span_to_identify = {}
    span_to_geonameID = {}
    span_to_text = {}
    conn = make_conn()
    countries = set()
    with open(full_ann_path, 'r') as infile:
        geoname_IDs = {}
        t_id_to_span = {}
        for line in infile: 
            contents = line.strip().split('\t')
            parts = contents[1].split()
            if line.startswith('T'): 
                t_id = contents[0]
                relation = parts[0]
                start = parts[1]
                end = parts[2] 
                text = contents[2]
                span_to_identify[(start, end)] = relation
                span_to_text[(start, end)] = text
                t_id_to_span[t_id] = (start, end)
            else: 
                t_id = parts[1]
                geonameID = contents[2].replace('https://www.geonames.org/', '').split('/')[0] 
                if not geonameID.isdigit(): continue
                geoname_IDs[t_id] = geonameID

        new_span_to_identify = {} # only evaluate on spans that have geonameID
        for t_id in t_id_to_span: 
            span = t_id_to_span[t_id]
            if t_id in geoname_IDs: 
                geonameID = geoname_IDs[t_id]
                span_to_geonameID[span] = geonameID
                new_span_to_identify[span] = span_to_identify[span]
                if new_span_to_identify[span] == 'Identify': 
                    res = get_entry_by_id(geonameID, conn)
                    countries.add(res['country_code3'])
    
    return countries, new_span_to_identify, span_to_geonameID, span_to_text

def interannotator_agreement(ann_path, iaa_mapping, root):
    '''
    A third annotator (me, unless the initial disagremeent involves me) resolves disagreements
    '''
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)
    identify1 = []
    identify2 = []
    geonameIDs1 = []
    geonameIDs2 = []
    disagreements_file = open(os.path.join(root, 'geo_disagree.csv'), 'w')
    writer = csv.writer(disagreements_file)
    for orig_name in d: 
        annotated = d[orig_name]
        if len(annotated) == 1: continue
        assert len(annotated) == 2
        annotated1 = annotated[0]
        annotated2 = annotated[1]
        countries1, span_to_identify1, span_to_geonameID1, span_to_text1 = get_annotations(os.path.join(ann_path, annotated1 + '.ann'))
        countries2, span_to_identify2, span_to_geonameID2, span_to_text2 = get_annotations(os.path.join(ann_path, annotated2 + '.ann'))
        spans = set(span_to_identify1.keys()) | set(span_to_identify2.keys())
        for span in spans: 
            if span in span_to_text1: 
                text = span_to_text1[span]
            else: 
                text = span_to_text2[span]
            identify1_bool = 0
            identify2_bool = 0
            if span in span_to_identify1 and span_to_identify1[span] == 'Identify': 
                identify1_bool = 1
            if span in span_to_identify2 and span_to_identify2[span] == 'Identify': 
                identify2_bool = 1
            identify1.append(identify1_bool)
            identify2.append(identify2_bool)
            geonameID1 = span_to_geonameID1.get(span, '')
            geonameID2 = span_to_geonameID2.get(span, '')
            geonameIDs1.append(geonameID1)
            geonameIDs2.append(geonameID2)
            # print out disagreements 
            if identify1_bool != identify2_bool: 
                print("Identify disagree:", span, annotated1, annotated2, text)
                writer.writerow(['identify', span, annotated1, annotated2, text])
            if geonameID1 != geonameID2: 
                print("GeonameID disagree:", span, annotated1, annotated2, text)
                writer.writerow(['geonameID', span, annotated1, annotated2, text])
    disagreements_file.close()
    print("Agreement for identify:", cohen_kappa_score(identify1, identify2))
    print("Agreement for geonameID:", cohen_kappa_score(geonameIDs1, geonameIDs2))
    
def get_gold_from_two(ann_path, annotated, tie_breakers): 
    '''
    Resolves disagreements between two files and unifies their annotations.
    '''
    annotated1 = annotated[0]
    annotated2 = annotated[1]
    _, span_to_identify1, span_to_geonameID1, span_to_text1 = get_annotations(os.path.join(ann_path, annotated1 + '.ann'))
    _, span_to_identify2, span_to_geonameID2, span_to_text2  = get_annotations(os.path.join(ann_path, annotated2 + '.ann'))
    spans = set(span_to_identify1.keys()) | set(span_to_identify2.keys())
    
    span_to_identify = {}
    span_to_geonameID = {}
    span_to_text = {}
    conn = make_conn()
    countries = set()
    for span in spans: 
        if span in span_to_text1: 
            span_to_text[span] = span_to_text1[span]
        else:
            span_to_text[span] = span_to_text2[span]
        geonameID1 = span_to_geonameID1.get(span, '')
        geonameID2 = span_to_geonameID2.get(span, '')
        if geonameID1 != geonameID2: 
            key = ('geonameID', annotated1, annotated2, span)
            assert key in tie_breakers
            breaker = tie_breakers[key]
            if breaker == annotated1 and geonameID1 != '': 
                span_to_geonameID[span] = geonameID1
            elif breaker == annotated2 and geonameID2 != '': 
                span_to_geonameID[span] = geonameID2
        elif geonameID1 != '': 
            span_to_geonameID[span] = geonameID1
            
        if span not in span_to_geonameID: continue

        identify1_bool = 'Not_Identify' # binarized, includes "Associated" 
        identify2_bool = 'Not_Identify'
        if span in span_to_identify1 and span_to_identify1[span] == 'Identify': 
            identify1_bool = 'Identify' 
        if span in span_to_identify2 and span_to_identify2[span] == 'Identify': 
            identify2_bool = 'Identify'

        if identify1_bool != identify2_bool: 
            key = ('identify', annotated1, annotated2, span)
            assert key in tie_breakers
            breaker = tie_breakers[key]
            if breaker == annotated1 and span in span_to_identify1: 
                span_to_identify[span] = identify1_bool
            elif breaker == annotated2 and span in span_to_identify2:  
                span_to_identify[span] = identify2_bool
        elif span in span_to_identify1: 
            span_to_identify[span] = identify1_bool
           
        if span in span_to_identify and span_to_identify[span] == 'Identify': 
            res = get_entry_by_id(span_to_geonameID[span], conn)
            countries.add(res['country_code3'])
            
    return countries, span_to_identify, span_to_geonameID, span_to_text

def load_tie_breakers(input_file): 
    # return (disagreement type, annotator1, annotator2, span) : winner}
    ret = {}
    with open(input_file, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            span = tuple(row['indices'].replace('(', '').replace(')', '').replace('\'', '').split(', '))
            ret[(row['disagreement'], row['annotator1'], row['annotator2'], span)] = row['tie_breaker']
    return ret

def evaluate(original_path, ann_path, iaa_mapping):
    with open(iaa_mapping, 'r') as infile: 
        d = json.load(infile)  
    tie_breakers = load_tie_breakers(os.path.join(root, 'geo_disagree_resolved.csv'))
    conn = make_conn()
        
    span_matches = 0
    retrieved_count = 0
    relevant_count = 0
    total_spans = 0
    correct_geonameID = 0
    correct_recalled_geonameID = 0
    correct_country = 0
    correct_recalled_country = 0
    country_retrieved = 0
    country_relevant = 0
    country_match = 0
    first_span_acc = 0
    freq_span_acc = 0
    has_country_preds = 0
    page_level_acc = 0
    for orig_name in d: 
        # predicted labels
        _, o_span_to_identify, o_span_to_geonameID, o_span_to_text = get_annotations(os.path.join(original_path, orig_name + '.ann'))
        annotated = d[orig_name]
        # gold labels
        if len(annotated) == 1: 
            a_countries, a_span_to_identify, a_span_to_geonameID, a_span_to_text = get_annotations(os.path.join(ann_path, annotated[0] + '.ann'))
        else: 
            a_countries, a_span_to_identify, a_span_to_geonameID, a_span_to_text = get_gold_from_two(ann_path, annotated, tie_breakers)
            
        retrieved = set(o_span_to_geonameID.keys())
        relevant = set(a_span_to_geonameID.keys())
        span_matches += len(retrieved & relevant)
        total_spans += len(retrieved | relevant)
        retrieved_count += len(retrieved)
        relevant_count += len(relevant)
        o_countries = set()
        
        min_span = (9999999, 9999999)
        min_country = None
        country_counts = Counter()
        correct_countries = set()
        
        for span in (retrieved | relevant): 
            o_geo = o_span_to_geonameID.get(span, '') # predicted
            a_geo = a_span_to_geonameID.get(span, '') # gold
            if o_geo == a_geo: 
                correct_geonameID += 1
            if o_geo and o_geo == a_geo: 
                correct_recalled_geonameID += 1

            if o_geo: 
                o_country = get_entry_by_id(o_geo, conn)['country_code3']
                o_countries.add(o_country)
                country_counts[o_country] += 1
            else: 
                o_country = None
            if a_geo: 
                a_country = get_entry_by_id(a_geo, conn)['country_code3']
            else: 
                a_country = None
                
            if int(span[0]) < int(min_span[0]): 
                min_span = span
                min_country = o_country
            
            if o_country == a_country: 
                correct_country += 1
                correct_countries.add(o_country)
            if o_country and o_country == a_country: 
                correct_recalled_country += 1
                
        country_retrieved += len(o_countries)
        country_relevant += len(a_countries)
        country_match += len(o_countries & a_countries)
        
        if country_counts: 
            has_country_preds += 1
            print(annotated, country_counts.most_common(5), a_countries)
        if min_country in a_countries: 
            first_span_acc += 1
        if country_counts and country_counts.most_common(1)[0][0] in a_countries: 
            freq_span_acc += 1
        if (len(retrieved) == 0 and len(relevant) == 0) or \
            (country_counts and country_counts.most_common(1)[0][0] in correct_countries): 
            page_level_acc += 1

    print("GeonameID Span Idx Match Precision:", span_matches / retrieved_count, "Recall:", span_matches / relevant_count)
    print("Identify Country Precision:", country_match / country_retrieved, "Recall:", country_match / country_relevant)
    print("GeonameID accuracy:", correct_geonameID / total_spans)
    print("GeonameID accuracy on recalled spans:", correct_recalled_geonameID / retrieved_count)
    print("Country accuracy:", correct_country / total_spans)
    print("Country accuracy on recalled spans:", correct_recalled_country / retrieved_count)
    print("Accuracy at the page-level:", page_level_acc / float(len(d)))
    print("Identify with first span country:", first_span_acc / has_country_preds)
    print("Identify with most frequent span country:", freq_span_acc / has_country_preds)

if __name__ == "__main__":
    root = '/home/lucyl/llm_social_identities/outputs/identity/'
    original_path = root + 'geo_annotate_assign/'
    ann_path = root + 'geo_annotate_done/'
    iaa_mapping = root + 'geo_eval_name2names.json'    
    
    #sanity_check_doubly_annotated(original_path, iaa_mapping)
    #sanity_check(original_path, ann_path)
    #interannotator_agreement(ann_path, iaa_mapping, root)
    evaluate(root + 'geo_annotate_original/', ann_path, iaa_mapping)
