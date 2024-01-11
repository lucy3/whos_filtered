"""
O*Net scraping
"""
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import urllib.request
import string
from tqdm import tqdm
import json
import openai
import csv
import numpy as np

#openai.api_key = os.environ["OPENAI_API_KEY"]

def scrape_job_families(): 
    '''
    Outputs a dictionary of job families to occupations
    '''
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    family_file = os.path.join(base_folder, 'onet_families.html')
    with open(family_file, 'r') as infile: 
        soup = BeautifulSoup(infile.read(), 'html.parser')
        table = soup.find_all("table")[0]
        rows = []
        for i, row in tqdm(enumerate(table.find_all('tr'))):
            links = []
            occ = None
            if i != 0:
                row_contents = []
                for j, el in enumerate(row.find_all('td')): 
                    if j == 1: 
                        occ = el.text.strip().replace('\nBright Outlook', '').strip()
                    row_contents.append(el.text.strip())
                    a = el.find_all('a')
                    for l in a: 
                        links.append(l['href'])
                rows.append(row_contents)
            if links: 
                url = links[0] # this is the one to the occ page
                occ = occ.translate(str.maketrans('', '', string.punctuation)).replace(' ', '-').lower()
                source = urllib.request.urlopen(url)
                text = source.read().decode()
                with open(os.path.join(base_folder, 'onet_pages/' + occ + '.html'), 'w') as outfile: 
                    outfile.write(text + '\n')
    occ_cats = defaultdict(list)
    for row in rows: 
        occ_title = row[1].replace('\nBright Outlook', '').strip()
        occ_cat = row[2]
        occ_cats[occ_cat].append(occ_title)
    
    with open(os.path.join(base_folder, 'onet_families.json'), 'w') as outfile: 
        json.dump(occ_cats, outfile)
    
def get_sample_jobs_and_salaries(): 
    '''
    Outputs a nested dictionary of job family -> occupation -> job titles
    Also outputs occupation -> salary
    '''
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'onet_families.json'), 'r') as infile: 
        occ_cats = json.load(infile)
        
    occ_jobs = defaultdict(list) # {hyphenated occupation title : [jobs]} 
    occ_salary = {}
    for f in tqdm(os.listdir(os.path.join(base_folder, 'onet_pages/'))): 
        occ_hyph = f.replace('.html', '')
        with open(os.path.join(base_folder, 'onet_pages/', f), 'r') as infile: 
            soup = BeautifulSoup(infile.read(), 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs: 
                text = p.text.strip()
                if text.startswith('Sample of reported job titles:'): 
                    if '; ' in text: 
                        job_list = text.split('\n')[1].split('; ')
                    else: 
                        job_list = text.split('\n')[1].split(', ')
                    for j in job_list: 
                        new_j = j.lower()
                        if '(' in new_j: 
                            paren_text = new_j[new_j.find("(")+1:new_j.rfind(")")]
                            new_j = new_j.replace('(' + paren_text + ')', '')
                            occ_jobs[occ_hyph].append(paren_text.strip())
                            occ_jobs[occ_hyph].append(new_j.strip())
                        else: 
                            occ_jobs[occ_hyph].append(new_j.strip())
            wages_div = soup.find("div", {"id": "WagesEmployment"})
            if not wages_div: 
                # no wage info
                continue
            dl_data = wages_div.find_all("dd")[0]
            wages = dl_data.text.split(', ')
            annual_salary = None
            for w in wages: 
                if w.endswith('annual'): 
                    annual_salary = int(w.replace('$', '').replace('+', '').replace(' annual', '').replace(',', ''))
            if not annual_salary: 
                print(f)
                continue
            occ_salary[occ_hyph] = annual_salary
            
    occ_hierarchy = defaultdict(dict)
    for cat in occ_cats: 
        occs = occ_cats[cat]
        for occ in occs: 
            occ_hyph = occ.translate(str.maketrans('', '', string.punctuation)).replace(' ', '-').lower()
            jobs = occ_jobs[occ_hyph]
            occ_hierarchy[cat][occ] = jobs

    with open(os.path.join(base_folder, 'onet_hierarchy.json'), 'w') as outfile: 
        json.dump(occ_hierarchy, outfile)
        
    with open(os.path.join(base_folder, 'onet_salary.json'), 'w') as outfile: 
        json.dump(occ_salary, outfile)
        
def query_openai(occ_title): 
    prompt = 'Split the following list of occupations and convert to singular: Watch and Clock Repairers\n\nAnswer: watch repairer, clock repairer \n\nSplit the following list of occupations and convert to singular: Plumbers, Pipefitters, and Steamfitters\n\nAnswer: plumber, pipefitter, steamfitter\n\nSplit the following list of occupations and convert to singular: '
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {
                  "role": "user",
                  "content": prompt + occ_title
                },
            ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response

def supplement_jobs_with_chatgpt(): 
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'onet_hierarchy.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    responses = {}
    for cat in tqdm(occ_hierarchy): 
        for occ in occ_hierarchy[cat]: 
            success = False
            while not success: 
                try: 
                    responses[occ] = query_openai(occ)
                    success = True
                except openai.error.APIError as e:
                    print(e)
                    continue
    
    with open(os.path.join(base_folder, 'chatgpt_onet.json'), 'w') as outfile: 
        json.dump(responses, outfile)

def supplement_jobs(rerun_chatgpt=False): 
    '''
    Take in the chatgpt responses and write out a csv containing
    agreements/disagreements with rule-based comma and "and" splitting approach. 
    
    Then, manually adjust based on disagreements. 
    '''
    if rerun_chatgpt: 
        supplement_jobs_with_chatgpt()
    
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'chatgpt_onet.json'), 'r') as infile: 
        chatgpt_responses = json.load(infile)
        
    with open(os.path.join(base_folder, 'onet_hierarchy.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    with open(os.path.join(base_folder, 'occ_title_cleanup.csv'), 'w') as outfile: 
        fieldnames = ['occ_title', 'chatgpt']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for cat in tqdm(occ_hierarchy): 
            for occ in occ_hierarchy[cat]: 
                title_list = []
                occs = occ.lower().strip().replace(' and ', ',').split(',')
                for job in occs: 
                    job = job.strip()
                    if not job: continue
                    if job.endswith('s'): 
                        job = job[:-1]
                    title_list.append(job)
                title_list = sorted(title_list)
                
                response = chatgpt_responses[occ]['choices'][0]['message']['content'].replace('Answer: ', '').lower().split(', ')
                response = sorted(response)
                
                if title_list == response: 
                    # everything chill
                    continue 
                    
                curr_list = set(occ_hierarchy[cat][occ])
                extra = set(response) - curr_list
                if extra: 
                    d = {}
                    d['occ_title'] = occ
                    d['chatgpt'] = ', '.join(response)
                    writer.writerow(d)

def polish_data(): 
    '''
    This function does two things: 
    1) adds supplemented job titles to onet hierarchy 
    2) remaps salaries so that job titles are mapped to salaries, and titles that map to multiple
    jobs (e.g. research scientist) take the average of its possible salaries. 
    '''
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    extra = {}
    with open(os.path.join(base_folder, 'occ_title_cleanup_done.csv'), 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            if row['custom'].strip(): 
                extra[row['occ_title']] = row['custom'].strip().split(', ')
            elif row['keep_chatgpt'].strip(): 
                extra[row['occ_title']] = row['chatgpt'].strip().split(', ')
                
    with open(os.path.join(base_folder, 'chatgpt_onet.json'), 'r') as infile: 
        chatgpt_responses = json.load(infile)
        
    with open(os.path.join(base_folder, 'onet_hierarchy.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
        
    code2occ = get_code2occ(base_folder)
    alternate_occ = defaultdict(list)
    with open(os.path.join(base_folder, 'onet_alternate_titles.txt'), 'r') as infile: 
        reader = csv.DictReader(infile, delimiter='\t') 
        for row in reader: 
            occ = code2occ[row['O*NET-SOC Code']]
            new_j = row['Alternate Title'].lower()
            if '(' in new_j: 
                paren_text = new_j[new_j.find("(")+1:new_j.rfind(")")]
                new_j = new_j.replace('(' + paren_text + ')', '')
                alternate_occ[occ].append(paren_text.strip())
                alternate_occ[occ].append(new_j.strip())
            else: 
                alternate_occ[occ].append(new_j.strip())
        
    occ_hierarchy_plus = defaultdict(dict)
    for cat in tqdm(occ_hierarchy): 
        for occ in occ_hierarchy[cat]: 
            job_list = set(occ_hierarchy[cat][occ])
            title_list = []
            occs = occ.lower().strip().replace(' and ', ',').split(',')
            for job in occs: 
                job = job.strip()
                if not job: continue
                if job.endswith('s'): 
                    job = job[:-1]
                title_list.append(job)
            title_list = sorted(title_list)

            response = chatgpt_responses[occ]['choices'][0]['message']['content'].replace('Answer: ', '').lower().split(', ')
            response = sorted(response)

            if title_list == response: 
                job_list.update([e for e in title_list if e != 'all other'])
            else: 
                curr_list = set(occ_hierarchy[cat][occ])
                more = set(response) - curr_list
                if more and occ in extra: 
                    job_list.update([e for e in extra[occ] if e != 'all other'])
                    
            if occ in alternate_occ: 
                job_list.update(alternate_occ[occ])
            
            occ_hierarchy_plus[cat][occ] = list(job_list)
            
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'w') as outfile: 
        json.dump(occ_hierarchy_plus, outfile)
        
    with open(os.path.join(base_folder, 'onet_salary.json'), 'r') as infile: 
        onet_salary = json.load(infile)
        
    job_salaries = defaultdict(list)
    for cat in occ_hierarchy_plus: 
        for occ in occ_hierarchy_plus[cat]: 
            occ_hyph = occ.translate(str.maketrans('', '', string.punctuation)).replace(' ', '-').lower()
            if occ_hyph not in onet_salary: continue
            salary = onet_salary[occ_hyph]
            
            for job in occ_hierarchy_plus[cat][occ]: 
                job_salaries[job].append(salary)
                
    job_salary = {}
    for job in job_salaries: 
        job_salary[job] = np.mean(job_salaries[job])
        
    with open(os.path.join(base_folder, 'job_salary.json'), 'w') as outfile: 
        json.dump(job_salary, outfile)
        
def get_code2occ(base_folder): 
    '''
    returns a dictionary mapping onet occupation codes to occupations
    '''
    family_file = os.path.join(base_folder, 'onet_families.html')
    with open(family_file, 'r') as infile: 
        soup = BeautifulSoup(infile.read(), 'html.parser')
        table = soup.find_all("table")[0]
        rows = []
        for i, row in tqdm(enumerate(table.find_all('tr'))):
            links = []
            occ = None
            if i != 0:
                row_contents = []
                for j, el in enumerate(row.find_all('td')): 
                    if j == 1: 
                        occ = el.text.strip().replace('\nBright Outlook', '').strip()
                    row_contents.append(el.text.strip())
                    a = el.find_all('a')
                    for l in a: 
                        links.append(l['href'])
                rows.append(row_contents)
    code2occ = {}
    for row in rows: 
        occ_title = row[1].replace('\nBright Outlook', '').strip()
        occ_code = row[0]
        code2occ[occ_code] = occ_title
        
    return code2occ
        
def get_prestige(): 
    '''
    Includes both occupation-level prestige and job-level prestige
    '''
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    code2occ = get_code2occ(base_folder)
    
    prestige_dict = {}
    with open(os.path.join(base_folder, 'OccupationalPrestigeRatings.csv'), 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in reader: 
            if row['ONET SOC 2018 Code'] == 'NA': continue
            code = row['ONET SOC 2018 Code'] + '.00'
            if row['OPR Job Rating'] == 'NA': continue
            prestige = float(row['OPR Job Rating'])
            if code in code2occ: 
                occ = code2occ[code]
                prestige_dict[occ] = prestige
    
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'r') as infile: 
        occ_hierarchy_plus = json.load(infile)
                
    job_prestiges = defaultdict(list)
    for cat in occ_hierarchy_plus: 
        for occ in occ_hierarchy_plus[cat]: 
            for job in occ_hierarchy_plus[cat][occ]: 
                if occ not in prestige_dict: continue
                prestige = prestige_dict[occ]
                job_prestiges[job].append(prestige)
        
    job_prest = {}
    for job in job_prestiges: 
        job_prest[job] = np.mean(job_prestiges[job])
    
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'occ_prestige.json'), 'w') as outfile: 
        json.dump(prestige_dict, outfile)
    with open(os.path.join(base_folder, 'job_prestige.json'), 'w') as outfile: 
        json.dump(job_prest, outfile)
        
def create_suffixes(): 
    '''
    dictionary of {last token : job titles containing last token}
    '''
    base_folder = '/home/lucyl/llm_social_identities/data/person_lists/'
    with open(os.path.join(base_folder, 'onet_hierarchy_plus.json'), 'r') as infile: 
        occ_hierarchy = json.load(infile)
    jobs = set()
    for cat in occ_hierarchy: 
        for occ in occ_hierarchy[cat]: 
            jobs.update(occ_hierarchy[cat][occ])
            
    last_to_job = defaultdict(list)
    for j in jobs: 
        last_word = j.split()[-1]
        last_to_job[last_word].append(j)
        
    with open(os.path.join(base_folder, 'last_to_job.json'), 'w') as outfile: 
        json.dump(last_to_job, outfile)
    
if __name__ == "__main__":
    #create_suffixes()
    #scrape_job_families()
    #get_sample_jobs_and_salaries()
    #supplement_jobs()
    #polish_data()
    get_prestige()
    
    