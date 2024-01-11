"""
Get lists of people from WordNet, Wikitionary, and job websites
"""

from nltk.corpus import wordnet as wn
from string import ascii_lowercase
import urllib
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import spacy

DATA = '/home/lucyl/llm_social_identities/data/'

def get_lemmas(word):
    person_words = set([' '.join(l.name().split('_')).lower() for l in word.lemmas()])
    for h in word.hyponyms():
        person_words.update(get_lemmas(h))
    return person_words

def get_wordnet(): 
    person = wn.synset('person.n.01')
    
    all_person_words = get_lemmas(person)
    
    with open(os.path.join(DATA, 'person_lists/wordnet.txt'), 'w') as outfile: 
        for person in all_person_words: 
            outfile.write(person + '\n')
            
def download_wiktionary(): 
    base_url = 'https://en.wiktionary.org/w/index.php?title=Category:en:People&from='
    for a in tqdm(ascii_lowercase):
        for b in ascii_lowercase:
            c = a + b
            source = urllib.request.urlopen(base_url + c.upper())
            text = source.read().decode()
            with open(os.path.join(DATA, 'wiktionary/' + c + '.html'), 'w') as outfile: 
                outfile.write(text + '\n')
    
def get_words_from_wiktionary(): 
    people_words = set()
    file_list = sorted(os.listdir(os.path.join(DATA, 'wiktionary/')))
    for i, f in enumerate(file_list): 
        with open(os.path.join(DATA, 'wiktionary/', f), 'r') as infile: 
            html_doc = infile.read()
            soup = BeautifulSoup(html_doc, 'html.parser')
            mydivs = soup.find_all("div", {"class": "mw-category-group"})
            has_overlap = False
            for div in mydivs: 
                uls = div.find('ul')
                texts = uls.text.split('\n')
                for text in texts: 
                    text = text.strip()
                    if text.startswith('en:'): continue
                    if text in people_words: 
                        has_overlap = True
                    people_words.add(text)
            if not f.startswith('aa') and not has_overlap: 
                # these are pages where we should manually wget the previous pages
                # rename the pages to [previous page letters]2.html
                print(f)
                print("Prev file exists:", os.path.exists(os.path.join(DATA, 'wiktionary/', file_list[i-1])))
    with open(os.path.join(DATA, 'person_lists/wiktionary.txt'), 'w') as outfile: 
        for person in people_words: 
            outfile.write(person + '\n')
            
def compare_lists(): 
    wordnet_words = set()
    with open(os.path.join(DATA, 'person_lists/wordnet.txt'), 'r') as infile: 
        for line in infile: 
            wordnet_words.add(line.strip().lower())
            
    wiktionary_words = set()
    with open(os.path.join(DATA, 'person_lists/wiktionary.txt'), 'r') as infile: 
        for line in infile: 
            wiktionary_words.add(line.strip().lower())
            
    print("Wordnet", len(wordnet_words))
    print("Wiktionary", len(wiktionary_words))
    print("Intersection:", len(wiktionary_words & wordnet_words))
    print("Union:", len(wiktionary_words | wordnet_words))
    
def get_ngrams(): 
    '''
    Combine the lists and bucket them
    by ngram length
    '''
    people_list = set()
#     with open(os.path.join(DATA, 'person_lists/wordnet.txt'), 'r') as infile: 
#         for line in infile: 
#             people_list.add(line.strip().lower())
            
    with open(os.path.join(DATA, 'person_lists/wiktionary.txt'), 'r') as infile: 
        for line in infile: 
            people_list.add(line.strip().lower())
            
    people_list = list(people_list)
            
    # Tokenize the people_list
    nlp = spacy.load("en_core_web_trf")
    bucket = defaultdict(list)
    for person in people_list: 
        doc = nlp.make_doc(person)
        bucket[len(doc)].append(person)
        
    total = 0
    for l in bucket: 
        print(l, len(bucket[l]))
        if l > 3: 
            print(bucket[l])
        else: 
            total += len(bucket[l])
    print("Unigram-bigram-trigram total:", total)
        
    with open(os.path.join(DATA, 'person_lists/ngram_buckets.json'), 'w') as outfile: 
        json.dump(bucket, outfile)

if __name__ == "__main__":
    #get_wordnet()
    #download_wiktionary()
    #get_words_from_wiktionary()
    #compare_lists()
    get_ngrams()