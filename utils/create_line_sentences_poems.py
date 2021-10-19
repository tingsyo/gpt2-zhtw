#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script processes corpus document and creates line-sentence documents.
'''
from __future__ import print_function
import logging
import re 
import os
import json
import argparse
from tqdm import tqdm

def scan_files(datapath):
    ''' Walk through the input directory to list all files for processing. '''
    import os
    urls = []
    for root, dirs, files in os.walk(datapath, topdown=False):
       for name in files:
            if name.startswith('poet'):
                urls.append(os.path.join(root, name))
    return(urls)


def process_zh_wiki_file(furl, converter, min_length=1):
    ''' Process the wikipedia data: each line contains an article in simplified Chinese as a dictionary. '''
    logging.debug(furl)
    output = []
    # 1. Read in lines
    with open(furl, 'r') as f:
        raw = f.readlines()
    # 2. Loop through a list of wikipedia documents
    for doc in raw:
        doc = eval(doc)
        text = doc['text'].split('\n')          # Separate document by line-breaks
        for p in text:                          # Loop through paragraphs
            if len(p) >= min_length:
                output.append(converter.convert(p))
    # Done
    return(output)

def process_poem_file(furl):
    ''' Process the poem data: list of json object is plain text. '''
    logging.debug(furl)
    # 1. Read in the list of json objects
    with open(furl, 'r') as f:
        raw = json.load(f)
    # 2. Loop through a list of json objects
    poems = [''.join(a['paragraphs']) for a in raw]
    # Done
    return(poems)



#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Create line sentence document for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing input documents.')
    parser.add_argument('--output', '-o', default='output', help='the directory for output txt files.')
    parser.add_argument('--min_length', default=4, type=int, help='the minimal tokens in an article to be included.')
    parser.add_argument('--output_size', default=1000, type=int, help='the number of sentences to included in one output file.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # 1. Scan for files
    poemfiles = scan_files(args.input)
    # 2. Loop through wiki-files
    num_input_files = len(poemfiles)
    logging.info("Total input files: "+str(num_input_files))
    output = None
    for i in tqdm(range(num_input_files)):
        tmp = process_poem_file(furl=poemfiles[i])
        logging.debug('Sentences in file '+poemfiles[i]+': '+str(len(tmp)))
        if output is None:
            output = tmp
        else:
            output += tmp
    # 3. Output by part
    if not os.path.exists(args.output):             # Check output path
        os.mkdir(args.output)
    num_poems = len(output)
    num_output_files = num_poems//args.output_size + 1
    logging.info("Output "+str(num_poems)+" poems into "+ str(num_output_files)+" files in "+args.output)
    for j in tqdm(range(num_output_files)):
        outfile = args.output+'/line_sentence_'+str(j).zfill(6)+'.txt'
        start_idx = j*args.output_size
        end_idx = min(len(output), j*args.output_size+args.output_size)
        tmp = output[start_idx:end_idx]
        with open(outfile, 'w') as f:
            f.write('\n'.join(tmp))
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()