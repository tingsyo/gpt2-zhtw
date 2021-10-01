#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script processes corpus document and creates line-sentence documents.
'''
from __future__ import print_function
import logging, os, argparse
import opencc
#from wikicorpus_with_punctuations import MyWikiCorpus
from gensim.corpora import WikiCorpus as MyWikiCorpus
''' This script converts wikipedia dumps (in XML) to txt with gensim.corpora.WikiCorpus '''

def tokenize(content):
    #override original method in wikicorpus.py
    return [token.encode('utf8') for token in content.split() 
           if len(token) <= 15 and not token.startswith('_')]

def tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list:
    return [token for token in text.split() if token_min_len <= len(token) <= token_max_len]
#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Create line sentence document for further processing.')
    parser.add_argument('--input', '-i', help='the directory containing unprocessed documents.')
    parser.add_argument('--output', '-o', default='output', help='the directory for output txt files.')
    parser.add_argument('--min_tokens', default=250, type=int, help='the minimal tokens in an article to be included.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # 1. Initialize WikiCorpus
    wiki_corpus = MyWikiCorpus(args.input, lemmatize=False, dictionary={}, lower=False, tokenizer_func=tokenizer_func, article_min_tokens=args.min_tokens)
    converter = opencc.OpenCC('s2tw.json')          # Chinese converter
    # For output
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # 2. Loop through downloaded wiki-dumps
    texts_num = 1
    space = " "
    for text in wiki_corpus.get_texts():
        article = space.join(text) + "\n"                   # Concatenate sentences by line break
        article_zhtw = converter.convert(article)           # Convert content to TW-Chinese
    # 3. Write processed article        
        outfile = os.path.join(args.output, 'article_{}.txt'.format(texts_num))
        with open(outfile,'w',encoding='utf-8') as output:
            output.write(article_zhtw)
        if texts_num % 10000 == 0:
            logging.info("已處理 %d 篇文章" % texts_num)    
        texts_num += 1
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()