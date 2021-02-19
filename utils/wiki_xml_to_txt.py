# -*- coding: utf-8 -*-
import logging, os, argparse
import opencc
from gensim.corpora import WikiCorpus

''' This script converts wikipedia dumps (in XML) to txt with gensim.corpora.WikiCorpus '''

#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', help='the wiki-dumps in xml.bz2')
    parser.add_argument('--output', '-o', default='output', help='the output txt file.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # 1. Initialize WikiCorpus
    wiki_corpus = WikiCorpus(args.input, dictionary={})
    texts_num = 0
    # 2. Proceed conversion
    converter = opencc.OpenCC('s2tw.json')
    with open(args.output,'w',encoding='utf-8') as output:
        for text in wiki_corpus.get_texts():
            article = ' '.join(text)
            article_zhtw = converter.convert(article) 
            output.write(article_zhtw + '\n')
            texts_num += 1
            if texts_num % 10000 == 0:
                logging.info("已處理 %d 篇文章" % texts_num)    
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()