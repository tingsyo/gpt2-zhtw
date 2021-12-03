#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script creates poem-like paragraphs starting with PTTGOSSIP data.

'''
from __future__ import print_function
import logging, os, argparse
from transformers import BertTokenizerFast, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import pickle
import json
import requests

STOP_WORDS = ['ETtoday','CrazyWinnie', 'ETTODAY', 'YAHOO新聞', '東森新聞', '-----']
BASE_URL='http://52.26.156.12:4001/get_json/?'

def retrieve_pttgossip(base_url=BASE_URL, data_date=None):
    if not data_date is None:
        date_string = '&item_date='+data_date
    else:
        data_string = ''
    #
    posts = requests.get(base_url+'item_type=posts'+data_string)
    tfreq = requests.get(base_url+'item_type=termscores'+data_string)
    # Convert to DataFrame
    posts = pd.DataFrame(eval(posts.text))
    tfreq = pd.DataFrame(eval(tfreq.text))
    return((posts, tfreq))


def article_to_sentences(article, min_length=5):
    ''' Parse an article into a list sentences. '''
    # Define sentence-break symbols
    bs = ['\n','，','。','；','！','？','「','」','.',':','（','）','／','　','~','：']
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


def generate_starting_sentence(st, min_sentences=10, base_url=BASE_URL, data_date=None):
    ''' Use the seeding information to create the starting sentence. '''
    posts, tfreq = retrieve_pttgossip(base_url=BASE_URL, data_date=data_date)
    #
    term_embeddings = st.encode(list(tfreq['term'].iloc[:N_TERMS]))
    term_weights = list(tfreq['score'].iloc[:N_TERMS])
    #
    sent = []
    sent_scores = []
    # Loop through articles
    for i in range(10):                             # Look at at most 10 articles
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
    sentence = ''.join(results['sentence'].iloc[:2])
    return(sentence)


def evaluate_tokens(tokens, model):
    embeddings = model.encode(tokens)
    return(embeddings)


def decode_generated_ids(generated, tokenizer):
    ''' Decode the ids generated by the language model. '''
    output=[]
    for i in range(10):
        text = tokenizer.decode(generated[i], skip_special_tokens= True)    # Decode the generated text
        text = text.replace(' ','')                                         # Remove spaces between tokens
        text = text.replace(',','，')
        output.append(text)
    return(output)

def generate_new_sentences(input, tokenizer, model, params):
    ''' Generate new sentences with specified model and tokenizer. '''
    # Parse seeding string
    input_ids = tokenizer.encode(input, return_tensors='pt')
    # Generate text
    generated = model.generate(input_ids, 
                            max_length=params['max_length'],  
                            num_return_sequences=params['num_return_sequences'],
                            no_repeat_ngram_size=params['no_repeat_ngram_size'],
                            repetition_penalty=params['repetition_penalty'],
                            length_penalty=params['length_penalty'],
                            top_p=params['top_p'],
                            temperature=params['temperature'],
                            top_k=params['top_k'],
                            do_sample=True,
                            early_stopping=True)
    # Decode
    output = decode_generated_ids(generated, tokenizer)
    # Done
    return(output)


def postprocess_generated_sentences(sentences, history_sentences, sent_transformer):
    ''' Post-process the generated paragraph. '''
    # Define sentence-break symbols
    bs = ['，','。','；','！','？','「','」']
    seed_sentence = history_sentences[-1]
    # Loop through all generated snetences
    svecs = []
    stokens = []
    for s in sentences:
        temp = s.replace(seed_sentence, '')     # Remove the seed sentence
        # Looking for sentence-break symbols
        idxs = [i for i, x in enumerate(temp) if x in bs]
        if len(idxs)>1:                         # Keep tokens before the fisrt break
            tokens = temp[idxs[0]+1:idxs[1]]
            logging.debug("Take the segment between the 1st and 2nd punchuations. "+str(len(idxs)))
            if tokens.strip()=='':
                logging.debug("Empty sentence, skip.")
                continue
            if tokens in history_sentences:
                logging.debug("Generated senytence already existed, skip.")
                continue
        #elif len(idxs)>0:
        #    tokens = tokens[:idxs[0]]
        else:                                   # Skip empty sentence
            logging.debug('The generated sentence is too short, skip it: '+s)
            continue
        svec = sent_transformer.encode(tokens)   # Calculate the sentence-embedding vectors of the tokens
        svecs.append({'sentence':tokens, 'embedding':svec})
    #
    return(svecs)


def select_next_sentence(candidates, embeddings, back_length=3):
    ''' Select the best candidate. '''
    scores = []
    for i in range(len(candidates)):
        score = 0
        logging.debug(candidates[i]['sentence'])
        emb_length = len(embeddings)
        if emb_length<back_length:
            seed_vec = embeddings[-1]
            score += np.dot(seed_vec, candidates[i]['embedding'])
        else:
            for j in range(emb_length, emb_length-back_length, -1):
                seed_vec = embeddings[j-1]
                weight = j-emb_length+back_length
                weight_sign = (weight%2)==1 and 1 or -1
                #logging.debug([j, weight, weight_sign])
                score += np.dot(seed_vec, candidates[i]['embedding'])*(weight)*(weight_sign)
        logging.debug(score)
        scores.append(score)
    return(candidates[scores.index(max(scores))])


def generate_poem(seed_sentence, model, tokenizer, st, params):
    ''' Generate a poem starting with seed_sentence '''
    output = []
    embeddings = []
    output.append(seed_sentence)
    seed_vec = evaluate_tokens(seed_sentence, st)
    embeddings.append(seed_vec)
    # Generate followed-up sentences
    for i in range(params['total_lines']):
        generated = generate_new_sentences(seed_sentence, tokenizer, model, params)
        candidates = postprocess_generated_sentences(generated, output, st)
        if len(candidates)>0:
            selected = select_next_sentence(candidates, embeddings)
        else:
            logging.debug('No, available generated sentences, skip empty step: '+str(i))
            continue
        output.append(selected['sentence'])
        embeddings.append(selected['embedding'])
        seed_vec = selected['embedding']
        seed_sentence = selected['sentence']
    # done
    poem = '\n'.join(output)
    return(poem)

#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Generate a poem with specified languag model and starting information.')
    parser.add_argument('--input', '-i', help='the input information for the starting sentence.')
    parser.add_argument('--config_file', '-c', default=None, help='the configuration file.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--random_seed', '-r', default=None, type=int, help='the seed for random numbers.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Prompt the setting
    logging.debug(args)
    # Default configuration
    TOKENIZER_PATH = '../model/test'
    MODEL_PATH = '../model/test'
    MODEL_TF = True
    WORD_EMBEDDING_PATH = 'distiluse-base-multilingual-cased-v2'
    GEN_PARAMS = {
        "max_length": 30,  
        "num_return_sequences": 10,
        "no_repeat_ngram_size": 2,
        "repetition_penalty": 1.5,
        "length_penalty": 1.0,
        "top_p": 0.92,
        "temperature": 0.85,
        "top_k": 16
    }
    # Load configuration file if specified
    if not args.config_file is None:
        conf = json.load(open(args.config_file, 'r'))
        TOKENIZER_PATH = conf['tokenizer_path']
        MODEL_PATH = conf['model_path']
        MODEL_TF = conf['model_tf']
        WORD_EMBEDDING_PATH = conf['word_embedding_path']
        GEN_PARAMS = conf['gen_params']
    # Initialize tokenizer and model
    logging.info("Loading tokenizer from "+TOKENIZER_PATH)
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    logging.info("Loading language model from "+MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, from_tf=eval(MODEL_TF))
    # Load word embeddings
    logging.info("Loading pre-trained sentence transformer from "+WORD_EMBEDDING_PATH)
    st = SentenceTransformer(WORD_EMBEDDING_PATH)
    # Generate random numbers
    np.random.seed(args.random_seed)            # Set random-state
    total_lines = np.random.randint(5,15)       # Define the total lines
    GEN_PARAMS['total_lines'] = total_lines
    # Generate starting sentence
    seed_sentence = generate_starting_sentence(st)
    logging.info('To generate '+str(total_lines)+' sentences starting with ['+seed_sentence+']')
    # Generate followed-up sentences
    output = generate_poem(seed_sentence, model, tokenizer, st, GEN_PARAMS)
    # done
    print(output)
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()