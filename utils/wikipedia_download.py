# -*- coding: utf-8 -*-
import wget
import os
import argparse

''' This script downloads wikipedia dumps for specified language (default: zh) '''

def download_wikidumps(lang, opath):
    origin = 'https://dumps.wikimedia.org/{}wiki/latest/{}wiki-latest-pages-articles.xml.bz2'.format(lang,lang)
    fname = opath+'/{}wiki-latest-pages-articles.xml.bz2'.format(lang)
    wget.download(origin, fname)
    return(1)

# Main
if __name__ == '__main__':
    ARGS_PARSER = argparse.ArgumentParser()
    ARGS_PARSER.add_argument('--lang', default='zh', type=str, help='language code to download from wikipedia corpus')
    ARGS_PARSER.add_argument('--outdir', default='./', type=str, help='output path to store the downloaded file')
    ARGS = ARGS_PARSER.parse_args()
    download_wikidumps(ARGS.lang, ARGS.outdir)
