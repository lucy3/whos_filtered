"""
C4 and Gopher rules
"""
import os
import json
from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
import gzip
import numpy as np
from urllib.parse import urlsplit
import pandas as pd

def gopher_rules(): 
    attribute_folder = '/home/lucyl/llm_social_identities/outputs/scores/gopher_rules' 
    output_name = 'gopher'
    prefix = 'cc__gopher_v1__'
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
        
    d = {
      "hn": [],
      "fraction_of_characters_in_duplicate_5grams": [],
      "fraction_of_characters_in_duplicate_lines": [],
      "fraction_of_characters_in_most_common_2grams": [],
      "fraction_of_duplicate_lines": [],
      "repetition_rule": [],
      "fraction_of_lines_ending_with_ellipsis": [],
      "ellipsis_rule": [],
      "fraction_of_lines_starting_with_bullet_point": [],
      "bullet_rule": [],
      "fraction_of_words_with_alpha_character": [],
      "alpha_rule": [],
      "mean_word_length": [],
      "wordlen_rule": [],
      "required_word_count": [],
      "stopword_rule": [],
      "symbol_to_word_ratio": [],
      "symbol_rule": [],
      "word_count": [],
      "doclen_rule": [],
      "keep": [],
    } 
    result = glob(attribute_folder + '/**/*.json.gz', recursive=True)
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[basename])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                d['hn'].append(hn)
                keep = True
                
                # does not contain between 50 and 100,000 words
                word_count = row['attributes'][prefix + 'word_count'][0][2]
                d['word_count'].append(word_count)
                if word_count > 50 and word_count < 100000:
                    d['doclen_rule'].append(True)
                else: 
                    d['doclen_rule'].append(False)
                    keep = False
                    
                # mean word length is outside the range of 3 to 10 characters
                mean_word_length = row['attributes'][prefix + 'mean_word_length'][0][2]
                d['mean_word_length'].append(mean_word_length)
                if mean_word_length >= 3 and mean_word_length <= 10: 
                    d['wordlen_rule'].append(True)
                else: 
                    d['wordlen_rule'].append(False)
                    keep = False
                
                # symbol-to-word ratio greater than 0.1 for either the hash symbol or the ellipsis
                symbol_to_word_ratio = row['attributes'][prefix + 'symbol_to_word_ratio'][0][2]
                d['symbol_to_word_ratio'].append(symbol_to_word_ratio)
                if symbol_to_word_ratio > 0.1: 
                    d['symbol_rule'].append(False)
                    keep = False
                else: 
                    d['symbol_rule'].append(True)

                # more than 90% of lines starting with a bullet point
                ## fraction_of_lines_starting_with_bullet_point
                fraction_of_lines_starting_with_bullet_point = row['attributes'][prefix + 'fraction_of_lines_starting_with_bullet_point'][0][2]
                d['fraction_of_lines_starting_with_bullet_point'].append(fraction_of_lines_starting_with_bullet_point)
                if fraction_of_lines_starting_with_bullet_point > 0.9: 
                    d['bullet_rule'].append(False)
                    keep = False
                else: 
                    d['bullet_rule'].append(True)

                # more than 30% ending with an ellipsis
                fraction_of_lines_ending_with_ellipsis = row['attributes'][prefix + 'fraction_of_lines_ending_with_ellipsis'][0][2]
                d['fraction_of_lines_ending_with_ellipsis'].append(fraction_of_lines_ending_with_ellipsis)
                if fraction_of_lines_ending_with_ellipsis > 0.3: 
                    d['ellipsis_rule'].append(False)
                    keep = False
                else: 
                    d['ellipsis_rule'].append(True)

                # < 80% of words in a document contain at least one alphabetic character
                fraction_of_words_with_alpha_character = row['attributes'][prefix + 'fraction_of_words_with_alpha_character'][0][2]
                d['fraction_of_words_with_alpha_character'].append(fraction_of_words_with_alpha_character)
                if fraction_of_words_with_alpha_character < 0.8: 
                    d['alpha_rule'].append(False)
                    keep = False
                else: 
                    d['alpha_rule'].append(True)

                # "stop word" filter, remove documents that do not contain at least 
                # two of the following English words: the, be, to, of, and, that, have, with
                required_word_count = row['attributes'][prefix + 'required_word_count'][0][2]
                d['required_word_count'].append(required_word_count)
                if required_word_count >= 2: 
                    d['stopword_rule'].append(True)
                else: 
                    d['stopword_rule'].append(False)
                    keep = False

                # filter out documents whose duplicate content surpasses any of the thresholds detailed in Table A1.
                repetition_rule = True
                ## fraction_of_characters_in_most_common_{2, 3, 4}grams
                if word_count >= 2: 
                    d['fraction_of_characters_in_most_common_2grams'].append(row['attributes'][prefix + 'fraction_of_characters_in_most_common_2grams'][0][2])
                else: 
                    d['fraction_of_characters_in_most_common_2grams'].append(0)
                if word_count >= 2 and row['attributes'][prefix + 'fraction_of_characters_in_most_common_2grams'][0][2] > 0.20: 
                    repetition_rule = False
                if word_count >= 3 and row['attributes'][prefix + 'fraction_of_characters_in_most_common_3grams'][0][2] > 0.18: 
                    repetition_rule = False
                if word_count >= 4 and row['attributes'][prefix + 'fraction_of_characters_in_most_common_4grams'][0][2] > 0.16: 
                    repetition_rule = False
                    
                ## fraction_of_characters_in_duplicate_{5, 6, 7, 8, 9, 10}grams
                if word_count >= 5: 
                    d['fraction_of_characters_in_duplicate_5grams'].append(row['attributes'][prefix + 'fraction_of_characters_in_duplicate_5grams'][0][2])
                else: 
                    d['fraction_of_characters_in_duplicate_5grams'].append(0)
                if word_count >= 5 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_5grams'][0][2] > 0.15: 
                    repetition_rule = False
                if word_count >= 6 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_6grams'][0][2] > 0.14: 
                    repetition_rule = False
                if word_count >= 7 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_7grams'][0][2] > 0.13: 
                    repetition_rule = False
                if word_count >= 8 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_8grams'][0][2] > 0.12: 
                    repetition_rule = False
                if word_count >= 9 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_9grams'][0][2] > 0.11: 
                    repetition_rule = False
                if word_count >= 10 and row['attributes'][prefix + 'fraction_of_characters_in_duplicate_10grams'][0][2] > 0.10: 
                    repetition_rule = False
                    
                ## fraction_of_duplicate_lines
                fraction_of_duplicate_lines = row['attributes'][prefix + 'fraction_of_duplicate_lines'][0][2]
                d['fraction_of_duplicate_lines'].append(fraction_of_duplicate_lines)
                if fraction_of_duplicate_lines > 0.3: 
                    repetition_rule = False
                    
                ## fraction_of_characters_in_duplicate_lines
                fraction_of_characters_in_duplicate_lines = row['attributes'][prefix + 'fraction_of_characters_in_duplicate_lines'][0][2]
                d['fraction_of_characters_in_duplicate_lines'].append(fraction_of_characters_in_duplicate_lines)
                if fraction_of_characters_in_duplicate_lines > 0.2: 
                    repetition_rule = False
                
                d['repetition_rule'].append(repetition_rule)
                if not repetition_rule: 
                    keep = False
                
                d['keep'].append(keep)
                
    out_folder = '/home/lucyl/llm_social_identities/outputs/scores/'
    outpath = os.path.join(out_folder, output_name + '.parquet')
    df = pd.DataFrame.from_dict(d)
    df.to_parquet(outpath)
    
def c4_rules(): 
    return
    attribute_folder = '/home/lucyl/llm_social_identities/outputs/scores/c4_rules' 
    output_name = 'c4'
    prefix = 'cc__c4_v2__'
    
    in_folder = '/home/lucyl/llm_social_identities/outputs/domains_with_about'
    with open(os.path.join(in_folder, 'one_page_per_hn.json'), 'r') as infile: 
        urls_per_basename = json.load(infile)
        
    d = {
      "hn": [],
      'lines_with_no_ending_punctuation': [],
#         has_naughty_word: bool = False
#         has_javascript: bool = False
#         has_lorem_ipsum: bool = False
#         has_curly_brace: bool = False
#         line_count: int = 0
#         character_count: int = 0
      "keep": [],
    } 
    result = glob(attribute_folder + '/**/*.json.gz', recursive=True)
    for filename in tqdm(sorted(result)): 
        basename = os.path.basename(filename)
        urls = set(urls_per_basename[basename])
        with gzip.open(filename, 'rt') as infile: 
            for line in infile: 
                row = json.loads(line)
                url = row['id']
                if url not in urls: continue
                u = urlsplit(url)
                hn = u.hostname
                d['hn'].append(hn)
                keep = True
                
                # we exclude line-level filters from analysis, e.g.
                # remove lines w/o terminal punctuation mark 
                # remove lines with the word Javascript
                # remove Wikipedia citation markers
                # remove any lines containing cookies or terms of use
                # deduplicate three-sentence spans throughout dataset
                # only retain lines that contained at least 5 words
                
                # remove any page with fewer than 3 sentences (Dolma doesn't include this?) 
                # remove any page that contained bad words
                # remove any page where the phrase “lorem ipsum” appeared.
                # remove any pages that contained a curly bracket.
                

if __name__ == '__main__':
    gopher_rules()
    #c4_rules()