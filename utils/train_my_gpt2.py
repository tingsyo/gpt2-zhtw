#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
This script trains my own GPT2.
'''
from __future__ import print_function
import logging
import os
import sys
import random
import argparse
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast, TFGPT2LMHeadModel, GPT2Config

myconfig = GPT2Config(
                    n_ctx=1024,
                    n_embd=768,
                    n_head=12,
                    n_layer=6,
                    n_positions=1024,
                    vocab_size=25129
            )

def dummy_loss(y_true, y_pred):
    ''' A dummy loss function for causal language model. '''
    return tf.reduce_mean(y_pred)

def initialize_gpt2(pretrained_path=None):
    ''' Model initialization. '''
    if pretrained_path is None:
        model = TFGPT2LMHeadModel(myconfig)
    else:
        model = TFGPT2LMHeadModel.from_pretrained(pretrained_path)
    # 
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
    return(model)


def scan_line_sentence_files(datapath):
    ''' Walk through the input directory to list all files for processing. '''
    import os
    urls = []
    for root, dirs, files in os.walk(datapath, topdown=False):
       for name in files:
            if name.startswith('line_sentence_') and name.endswith('.txt'):
                urls.append(os.path.join(root, name))
    return(urls)

def process_line_sentence_file(furl, tokenizer):
    ''' Read the line-sentence text file and create tokenized dataset. '''
    # Read file
    with open(furl, 'r') as f:
        sentences = f.readlines()
    # Tokenization
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='np')
    # Create  dataset
    inputs, labels = [], []
    for token in tokens['input_ids']:
        inputs.append(token[:-1])
        labels.append(token[1:])
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    return(dataset)

def data_generator(data_files, tokenizer, batch_size=16, buffer_size=10000):
    '''  '''
    # Shuffle data_files
    file_ordering = np.random.permutation(len(data_files))
    for file_idx in file_ordering:
        dataset = process_line_sentence_file(data_files[file_idx], tokenizer)
        sample_ordering = np.random.permutation(len(dataset))
        for sample_idx in sample_ordering:
            example = list(dataset)[int(sample_idx)]
            yield example[0], example[1]
    return



#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Train my own GPT-2 with line-sentence data.')
    parser.add_argument('--input', '-i', help='the directory containing input documents.')
    parser.add_argument('--model_path', '-m', help='the path to store trained model.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    parser.add_argument('--epochs', '-e', default=3, help='epochs of training.')
    parser.add_argument('--batch_size', '-b', default=16, help='batch-size of training.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)
    # 1. Scan for files
    data_files = scan_line_sentence_files(args.input)
    num_input_files = len(data_files)
    logging.info("Total training files: "+str(num_input_files))
    # 2. Initialize model
    model = initialize_gpt2(args.model_path)
    tokenizer = BertTokenizerFast.from_pretrained('../model/tokenizer/')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.model_path+'/checkpoint',
        save_weights_only=True,
        monitor='loss',
        save_best_only=True)
    # 3. Prepare data
    dg = data_generator(data_files, tokenizer)
    # 4. Train model
    TOTAL_SENTENCES = len(data_files)*1000
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    history = model.fit(dg, epochs=EPOCHS, batch_size=BATCH_SIZE, steps_per_epoch=(TOTAL_SENTENCES//BATCH_SIZE)+1)
    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()
