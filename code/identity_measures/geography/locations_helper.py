"""
Analyze the output of geoparsed locations
"""

import os
from tqdm import tqdm
import json
from urllib.parse import urlsplit
import re
import argparse
import spacy
from collections import Counter, defaultdict
import tldextract
from glob import glob
import gzip
import csv

def get_coverage(): 
    spacy_folder = '/home/lucyl/llm_social_identities/outputs/identity/spacy_output/'
    entity_files = os.listdir(spacy_folder)
    has_gpe_count = 0
    total_count = 0
    for f in tqdm(entity_files): 
        with open(os.path.join(spacy_folder, f), 'r') as infile: 
            short_name = f.split("/")[-1].replace(".json.gz", "")
            for line in infile: 
                row = json.loads(line)
                ents = row['ents']
                has_gpe = False
                for ent in ents: 
                    if ent[1] in ['GPE', 'LOC']: 
                        has_gpe = True
                        break
                if has_gpe: 
                    has_gpe_count += 1
                total_count += 1
    print("Has GPE", has_gpe_count)
    print("Total", total_count)
    print("Percent coverage:", has_gpe_count*100/total_count)
    
def get_coverage_save_country(): 
    geoparse_folder = '/home/lucyl/llm_social_identities/outputs/identity/geoparse'
    finished = set()
    for filename in os.listdir(geoparse_folder): 
        if filename.endswith('.done'): 
            short_name = filename.split('/')[-1].replace('.done', '')
            finished.add(short_name)
    total_count = 0
    has_country = 0
    countries = Counter()
    hn_to_country = {}
    for filename in tqdm(os.listdir(geoparse_folder)): 
        total_in_file = 0
        if filename.endswith('.done'): continue
        short_name = filename.split('/')[-1].replace('.jsonl', '')
        if short_name not in finished: continue
        with open(os.path.join(geoparse_folder, filename), 'r') as infile: 
            for line in infile: 
                d = json.loads(line)
                url = d['url']
                u = urlsplit(url)
                hn = u.hostname
                total_count += 1
                found_countries = []
                for sent_idx in d: 
                    if sent_idx == 'url': continue
                    out = d[sent_idx]
                    if 'geolocated_ents' in out: 
                        for ent in out['geolocated_ents']: 
                            if 'country_code3' in ent and ent['country_code3'] != '' and ent['country_code3'] != 'NA': 
                                found_countries.append(ent['country_code3'])
                if found_countries: 
                    has_country += 1
                    total_in_file += 1
                    country_counts = Counter(found_countries)
                    max_count = 0
                    max_country = ''
                    for i, country in enumerate(found_countries): 
                        if country_counts[country] > max_count: 
                            max_count = country_counts[country]
                            max_country = country
                    hn_to_country[hn] = max_country
                    countries[max_country] += 1
        assert total_in_file > 0
        
    with open('/home/lucyl/llm_social_identities/outputs/identity/url_to_country.json', 'w') as outfile: 
        json.dump(hn_to_country, outfile)
    print("TOTAL", total_count)
    print(countries.most_common(100))
    print("Has country:", has_country)
    print("Has country fraction:", has_country / total_count)
    
def check_output(): 
    about_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    geoparse_folder = '/home/lucyl/llm_social_identities/outputs/identity/geoparse'
    finished = set()
    for filename in os.listdir(geoparse_folder): 
        if filename.endswith('.done'): 
            short_name = filename.split('/')[-1].replace('.done', '')
            finished.add(short_name)
    
    result = glob(about_path + "/**/*.json.gz", recursive=True)
    for filename in result: 
        short_name = filename.split('/')[-1].replace('.json.gz', '')
        if short_name in finished: 
            total = 0
            with gzip.open(filename, "rt") as infile:
                for line in infile:
                    total += 1
                    
            other_total = 0
            with open(os.path.join(geoparse_folder, short_name + '.jsonl'), 'r') as infile: 
                for line in infile: 
                    other_total += 1
                    
            print(total, other_total)
            
def organize_metadata(): 
    '''
    This outputs a csv with the following columns: 
    country code, name, most recent GDP in USD, continental region, subregion, anglophone status
    '''
    data_path = '/home/lucyl/llm_social_identities/data/countries/'
    code_anglophone = {}
    anglophone_path = os.path.join(data_path, 'anglophone_wikipedia.csv')
    with open(anglophone_path, 'r') as infile: 
        reader = csv.DictReader(infile) 
        for row in reader: 
            code = row['ISO Code']
            status = 'Official, not primary'
            if 'Yes' in row['Primary language?']: 
                status = 'Official, primary'
            if 'Yes' in row['Core anglophone? ']: 
                status = 'Core anglophone'
            code_anglophone[code] = status
            
    code_path = os.path.join(data_path, 'country_codes.csv')
    code_country = {}
    with open(code_path, 'r') as infile: 
        reader = csv.DictReader(infile) 
        for row in reader: 
            name = row['English short name']
            name = re.sub("[\(\[].*?[\)\]]", "", name)
            name = name.replace(', State of', '').replace(', the United Republic of', '')
            code_country[row['Alpha-3 code']] = name
          
    code_continent = {}
    continent_path = os.path.join(data_path, 'un_continents.csv')
    with open(continent_path, 'r') as infile: 
        reader = csv.DictReader(infile) 
        for row in reader: 
            code_continent[row['ISO-alpha3 code']] = (row['Continental regions'], row['Subregions'])

    gdp_path = os.path.join(data_path, 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5728855.csv')
    code_gdp = {}
    with open(gdp_path, 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            code = row['Country Code']
            start_year = 2022
            for i in range(20):
                if row[str(start_year)].strip() != "": 
                    gdp = float(row[str(start_year)])
                    break
                start_year -= 1
            code_gdp[code] = gdp
            
    with open(os.path.join(data_path, 'metadata.csv'), 'w') as csvfile:
        fieldnames = ['Code', 'Country', 'English status', 'Continental region', 'Subregion', 'GDP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for code in code_country: 
            country = code_country[code]
            english_status = 'Other'
            if code in code_anglophone: 
                english_status = code_anglophone[code]
            region = 'Unknown'
            subregion = 'Unknown'
            if code in code_continent: 
                region, subregion = code_continent[code]
                if subregion in set(['Micronesia', 'Melanesia', 'Polynesia']): 
                    subregion = 'Pacific Islands'
            gdp = 'Unknown'
            if code in code_gdp: 
                gdp = code_gdp[code]
            writer.writerow({'Code': code, 
                             'Country': country,
                             'English status': english_status,
                             'Continental region': region,
                             'Subregion': subregion,
                             'GDP': gdp,
                            })
            
def url_extensions(): 
    '''
    Coverage
    '''
    country_url_codes = set()
    data_path = '/home/lucyl/llm_social_identities/data/countries/'
    with open(os.path.join(data_path, 'url_extensions.csv'), 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            country_url_codes.add(row['ccTLD'])
    
    about_path = '/net/nfs.cirrascale/allennlp/lucyl/cc_data/cc_bios_v1'
    
    result = glob(about_path + "/**/*.json.gz", recursive=True)
    total = 0
    found = 0
    for filename in tqdm(result): 
        short_name = filename.split('/')[-1].replace('.json.gz', '')
        with gzip.open(filename, "rt") as infile:
            for line in infile:
                d = json.loads(line)
                url = d['id']
                u = urlsplit(url)
                hn = u.hostname
                total += 1
                suffix = '.' + tldextract.extract(hn).suffix
                if suffix in country_url_codes: 
                    found += 1
                    
    print("Percent with country code extensions:", found/total)
    
if __name__ == "__main__":
    #get_coverage()
    get_coverage_save_country()
    #check_output()
    #organize_metadata()
    #url_extensions()