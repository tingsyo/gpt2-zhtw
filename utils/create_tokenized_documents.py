#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script processes corpus document and creates line-sentence documents.
'''
from __future__ import print_function
import logging, os, argparse
import opencc
from tqdm import tqdm
import numpy as np

def scan_wiki_files(datapath):
    ''' Walk through the input directory to list all files for processing. '''
    import os
    urls = []
    for root, dirs, files in os.walk(datapath, topdown=False):
       for name in files:
            if name.startswith('wiki_'):
                urls.append(os.path.join(root, name))
    return(urls)


def tokenize_text(text, tokenizer):
    ''' Tokenize the text with specified tokenizer. '''
    block_size = tokenizer.model_max_length
    if len(text)<block_size:
        results = tokenizer.encode(text)
    else:
        results = tokenizer.encode(text[0:block_size])
        for i in range(1, len(text)-block_size+1, block_size):
            end = min(i+block_size, len(text))
            results.append(tokenizer.encode(text[i:end]))
    return(results)


def process_zh_wiki_file(furl, converter, tokenizer, min_length=1):
    ''' Process the wikipedia data: each line contains an article in simplified Chinese as a dictionary. '''
    logging.debug(furl)
    output = None
    # 1. Read in lines
    with open(furl, 'r') as f:
        raw = f.readlines()
    # 2. Loop through a list of wikipedia documents
    for doc in raw:
        doc = eval(doc)
        text = doc['text']
        if len(text) >= min_length:
            logging.debug('Encodeing text of length '+str(len(text)))
            text = converter.convert(text)
            tmp = tokenized_text(text, tokenizer)
            if output is None:
                output = tmp
            else:
                output += tmp
    # Done
    return(output)



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
    wikifiles = scan_wiki_files(args.input)
    # 2. Loop through wiki-files
    num_input_files = len(wikifiles)
    logging.info("Total input files: "+str(num_input_files))
    converter = opencc.OpenCC('s2tw.json')          # Chinese converter
    output = None
    for i in tqdm(range(num_input_files)):
        tmp = process_zh_wiki_file(furl=wikifiles[i], converter=converter, min_length=args.min_length)
        logging.debug('Sentences in file '+wikifiles[i]+': '+str(len(tmp)))
        if output is None:
            output = tmp
        else:
            output += tmp
    # 3. Output by part
    if not os.path.exists(args.output):             # Check output path
        os.mkdir(args.output)
    num_sentences = len(output)
    num_output_files = num_sentences//args.output_size + 1
    logging.info("Output "+str(num_sentences)+" sentences into "+ str(num_output_files)+" files in "+args.output)
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