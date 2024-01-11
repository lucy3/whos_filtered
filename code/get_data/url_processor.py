import logging
import multiprocessing
from contextlib import ExitStack
from queue import Queue
from typing import Dict
import zstandard
import gzip
import re
import os
from urllib.parse import urlsplit
from typing import Optional, List
from bloomfilter import BloomFilter
from collections import Counter, defaultdict

import json
from smashed.utils.io_utils import open_file_for_write, open_file_for_read
from smashed.utils.io_utils.io_wrappers import ReadTextIO

from ai2_llm_filters.core_tools.parallel import BaseParallelProcessor

def gzip_open(file, mode, **open_kwargs):
    return gzip.open(filename=file, mode=mode, **open_kwargs)

class CommonCrawlUrlProcessor(BaseParallelProcessor):
    @classmethod
    def get_logger(cls) -> logging.Logger:
        return logging.getLogger(cls.__name__)

    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        documents: int = 0,
        found_docs: int = 0,
    ) -> Dict[str, int]:
        """We override this method to specify which units we want to keep track of in a progress bar.
        Their default value must be zero."""

        # we call the super method to increment the progress bar
        return super().increment_progressbar(queue, files=files, documents=documents, found_docs=found_docs)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: "Queue",
        keywords: set = None,
        keep_path: str = None,
        bloomf: "BloomFilter" = None,
        reservoir: bool = False,
    ):
        """
        There are four uses for this function:
        - if keywords, then match based on presence of keyword as a component of the url
        - if keep_path and reservoir = False, there should be a file
        containing all of the urls we want to keep, one per line
        - if keep_path and reservoir = True, there should be a file containing a json
        of hostnames and how many times to sample per hostname
        - if bloomf, then filter based on it
        """
        assert keywords or keep_path or bloomf # we need to filter URLs based on *something*
        if keywords:
            assert not keep_path and not bloomf
            assert not reservoir
        if keep_path:
            assert not keywords and not bloomf
            if reservoir: 
                basename = source_path.split('/')[-1].replace('.json.gz', '')
                with open(os.path.join(keep_path, basename), 'r') as infile:
                    reservoir = json.load(infile)
            else: 
                about_urls = set()
                basename = source_path.split('/')[-1].replace('.json.gz', '')
                with open(os.path.join(keep_path, basename), 'r') as infile:
                    for line in infile:
                        about_urls.add(line.strip())
        if bloomf:
            assert not reservoir
            assert not keywords and not keep_path

        # interval at which to update the progress bar; will double if it gets
        # too full
        update_interval = 1

        # running document count; gets reset every time we update the progress
        # bar
        docs_cnt = 0
        found_docs_cnt = 0

        if bloomf:
            hostname_counts = defaultdict(Counter)
            
        if reservoir: 
            assert keep_path
            hn_idx = Counter()

        with ExitStack() as stack:
            # open each file for reading and writing. We use open_file_for_read to handle s3 paths and
            # download the file locally if needed, while gzip.open is used to
            # read and write gzipped files.
            if keywords or bloomf:
                # reading bytes from s3
                in_stream = stack.enter_context(open_file_for_read(source_path, "rb", open_fn=gzip_open))
            if keep_path:
                # reading written text
                in_stream = stack.enter_context(open_file_for_read(source_path, "rt", open_fn=gzip_open))
            out_stream = stack.enter_context(open_file_for_write(destination_path, 'wt', open_fn=gzip_open))

            for line in in_stream:
                row = json.loads(line)
                url = row['id']
                docs_cnt += 1

                if keywords:
                    parts = list(filter(None, re.split("[/.]+", url)))
                    if keywords & set(parts):
                        # write the output to the output file
                        out_stream.write(json.dumps(row, sort_keys=True) + '\n')
                        found_docs_cnt += 1
                if keep_path:
                    u = urlsplit(url)
                    hn = u.hostname
                    if reservoir:
                        if len(row['text']) < 100 or hn not in reservoir: continue
                        if hn_idx[hn] in reservoir[hn]: 
                            out_stream.write(json.dumps(row, sort_keys=True) + '\n')
                            found_docs_cnt += 1
                        hn_idx[hn] += 1
                    else: 
                        if url in about_urls:
                            out_stream.write(json.dumps(row, sort_keys=True) + '\n')
                            found_docs_cnt += 1

                if bloomf:
                    hostname_counts['all']['ALL'] += 1
                    if len(row['text']) >= 100:
                        hostname_counts['long']['ALL'] += 1
                    u = urlsplit(url)
                    hn = u.hostname
                    if hn and bloomf.check(hn):
                        found_docs_cnt += 1
                        hostname_counts['all'][hn] += 1
                        if len(row['text']) >= 100:
                            hostname_counts['long'][hn] += 1

                if docs_cnt % update_interval == 0:
                    # update the progress bar every 1000 documents to prevent
                    # buffering
                    cls.increment_progressbar(queue, documents=docs_cnt, found_docs=found_docs_cnt)
                    docs_cnt = 0
                    found_docs_cnt = 0

                    if queue.qsize() >= multiprocessing.cpu_count():
                        # double the update interval if the queue is full
                        update_interval *= 2

            if bloomf: 
                out_stream.write(json.dumps(hostname_counts) + '\n')

        # increment the files progress bar
        cls.increment_progressbar(queue, files=1, documents=docs_cnt, found_docs=found_docs_cnt)


def crawl_for_potential_about():
    '''
    Finds any url that
    contains the listed keywords
    @input: 
    - Common Crawl
    @output: 
    - Potential about pages in cc_bios_v0
    '''
    p = CommonCrawlUrlProcessor(
        source_prefix="s3://ai2-llm/pretraining-data/sources/common-crawl/v1/documents/", # this can be either S3 or local
        destination_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_bios_v0",   # this can be either S3 or local
        metadata_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_bios_v0_meta",    # doesn't redo ones that are already processed
        ignore_existing=False,   # if you set this to true, it redoes everything
        num_processes= multiprocessing.cpu_count() // 2,    # increase for more parallelism
        debug=False, # set to debug to add breakpoints
    )
    p(keywords=set(['about-me', 'about', 'about-us', 'bio']))

def crawl_for_about_pages():
    '''
    Retrieves about me pages that we
    wish to keep (cleaned up from crawl_for_potential_about()
    
    This is run after running phase1() in website_expansion.py
    '''
    p = CommonCrawlUrlProcessor(
        source_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_bios_v0/", # this can be either S3 or local
        destination_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1",   # this can be either S3 or local
        metadata_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_bios_v1_meta",    # doesn't redo ones that are already processed
        ignore_existing=False,   # if you set this to true, it redoes everything
        num_processes= multiprocessing.cpu_count() // 2,    # increase for more parallelism
        debug=False, # set to debug to add breakpoints
    )
    p(keep_path='/home/lucyl/llm_social_identities/outputs/domains_with_about/to_keep_per_split')

def count_target_hostnames():
    '''
    Counts number of pages per split total and number
    that are likely associated with a target hostname.
    Output: {'total': {ALL: int, hostname1: int, hostname2: int, ...},
            'long_enough': {ALL: int, hostname1: int, hostname2: int, ...},
        }
    '''
    with open('/home/lucyl/llm_social_identities/outputs/domains_with_about/domain_to_one_abouts.json', 'r') as infile:
        single_map = Counter(json.load(infile))

    p = 0.05
    bloomf = BloomFilter(len(single_map), p)
    for key in single_map:
        bloomf.add(key)

    p = CommonCrawlUrlProcessor(
        source_prefix="s3://ai2-llm/pretraining-data/sources/common-crawl/v1/documents/", # this can be either S3 or local
        destination_prefix="/net/nfs/allennlp/lucyl/cc_data/hostname_counts",   # this can be either S3 or local
        metadata_prefix="/net/nfs/allennlp/lucyl/cc_data/hostname_counts_meta",    # doesn't redo ones that are already processed
        ignore_existing=False,   # if you set this to true, it redoes everything
        num_processes= multiprocessing.cpu_count()*2 // 3,    # increase for more parallelism
        debug=False, # set to debug to add breakpoints
    )
    p(bloomf=bloomf)
    
def retrieve_sample_pages(): 
    '''
    Retrieve a sample of each host domain
    '''
    p = CommonCrawlUrlProcessor(
        source_prefix="s3://ai2-llm/pretraining-data/sources/common-crawl/v1/documents/", # this can be either S3 or local
        destination_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_sample",   # this can be either S3 or local
        metadata_prefix="/net/nfs/allennlp/lucyl/cc_data/cc_sample_meta",    # doesn't redo ones that are already processed
        ignore_existing=False,   # if you set this to true, it redoes everything
        num_processes= multiprocessing.cpu_count()*2 // 3,    # increase for more parallelism
        debug=False, # set to debug to add breakpoints
    )
    p(keep_path='/home/lucyl/llm_social_identities/outputs/domains_with_about/reservoir', reservoir=True)

if __name__ == "__main__":
    #crawl_for_potential_about()
    #crawl_for_about_pages()
    #count_target_hostnames()
    retrieve_sample_pages()
