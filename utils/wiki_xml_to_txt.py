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
    parser.add_argument('--min_tokens', default=250, type=int, help='the minimal tokens in an article to be included.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # 1. Initialize WikiCorpus
    wiki_corpus = WikiCorpus(args.input, dictionary={}, article_min_tokens=args.min_tokens)
    converter = opencc.OpenCC('s2tw.json')          # Chinese converter
    # For output
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    # 2. Loop through downloaded wiki-dumps
    texts_num = 1
    for text in wiki_corpus.get_texts():
        article = '\n'.join(text)                   # Concatenate sentences by line break
        article_zhtw = converter.convert(article)   # Convert content to TW-Chinese
    # 3. Write processed article        
        outfile = os.path.join(store_path, 'article_{}.txt'.format(texts_num))
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