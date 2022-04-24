#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script retrieves processed PTTGOSSIP data with API.
'''

from __future__ import print_function
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# Parameters
BASE_URL = 'http://52.26.156.12:4001/get_json/?'
TODAY = datetime.today()
DATA_DATE= datetime.strftime(TODAY-timedelta(days=1), '%Y-%m-%d')
STOP_WORDS = ['ETtoday','CrazyWinnie', 'ETTODAY', 'YAHOO新聞', 'Yhoo!新聞',
              '東森新聞','CTWANT','完整新聞標題','完整新聞內文', '-----']

# Functions
def retrieve_pttgossip(base_url=BASE_URL, data_date=None):
    if not data_date is None:
        date_string = '&item_date='+data_date
    else:
        date_string = ''
    #
    posts = requests.get(base_url+'item_type=posts'+date_string)
    tfreq = requests.get(base_url+'item_type=termscores'+date_string)
    # Convert to DataFrame
    posts = pd.DataFrame(eval(posts.text))
    tfreq = pd.DataFrame(eval(tfreq.text))
    return((posts, tfreq))

def retrieve_local_pttgossip(base_url='/home/tsyo/dail_poem/PTTGOSSIP_POSTS/', data_date=None):
    if not data_date is None:
        date_string = '&item_date='+data_date
    else:
        date_string = ''
    # Read local csv
    posts = pd.read_csv(base_url+date_string+'_posts.csv')
    tfreq = pd.read_csv(base_url+date_string+'_termscores.csv')
    return((posts, tfreq))

def article_to_sentences(article, min_length=5):
    ''' Parse an article into a list sentences. '''
    # Define sentence-break symbols
    bs = ['\n','，','。','；','！','？','「','」','.',':','（','）','／','　','~','：','《','》','、']
    # Loop through the article character-by-character
    sentences = []
    tmp = []
    for char in article:
        if not char in bs:
            tmp.append(char)
        else:
            if len(tmp)>=min_length:
                sentences.append(''.join(tmp).strip())
            tmp = []
    return(sentences)

def evaluate_sentence_embedding(se, term_embeddings, term_weights=None):
    ''' Evaluate one embedding vector against a list of embeddings (with weights). '''
    # Assign equal weights if not specified
    if term_weights is None:
        term_weights = np.ones(len(term_embeddings))
    # Start weighted averaging
    score = 0
    for i in range(len(term_embeddings)):
        score+=np.dot(se, term_embeddings[i])*term_weights[i]
    score = score/sum(term_weights)
    return(score)

def generate_starting_sentences(st, posts, tfreq, min_sentences=50, stop_words=STOP_WORDS):
    ''' Use the seeding information to create the starting sentence. '''
    term_embeddings = st.encode(list(tfreq['term'].iloc[:min_sentences]))
    term_weights = list(tfreq['score'].iloc[:min_sentences])
    #
    sent = []
    sent_scores = []
    # Loop through articles
    num_of_articles = posts.shape[0]
    for i in range(min(10, num_of_articles)):       # Look at at most 10 articles
        article = posts['content'].iloc[i]
        sentences = article_to_sentences(article)
        # Loop through sentences
        for s in sentences:
            if not s in stop_words:
                se = st.encode(s)
                sent.append(s)
                sent_scores.append(evaluate_sentence_embedding(se, term_embeddings, term_weights))
        # Check total number of sentences
        if len(sent_scores)>=min_sentences:
            break
    # Organize results
    results = pd.DataFrame({'sentence':sent, 'score':sent_scores})
    results = results.sort_values('score', ascending=False).reset_index(drop=True)
    logging.debug(results)
    return(results)


# Main script
logging.basicConfig(level=logging.DEBUG)
# Retrieve data with API
posts, tfreq = retrieve_local_pttgossip(base_url=BASE_URL, data_date=DATA_DATE)
posts.to_csv('posts.csv', index=False)
tfreq.to_csv('tfreq.csv', index=False)
# Load sentence transformer model
ST_PATH="../model/distiluse-base-multilingual-cased-v2/"
st = SentenceTransformer(ST_PATH)
# Generate canidate sentences
seed_sentences = generate_starting_sentences(st, posts, tfreq)
seed_sentences.to_csv('seed_sentences.csv', index=False)
