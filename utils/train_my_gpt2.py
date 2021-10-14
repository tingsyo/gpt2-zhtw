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
                    vocab_size=25129,
                    use_cache=True,
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
    # Tokenization with tokenizer.encode()
    block_size = tokenizer.model_max_length
    examples = []
    for sentence in sentences:
        if len(sentence)<=block_size: 
            examples.append(tokenizer.encode(sentence))
        else:                           # Truncate in block of block_size
            #logging.debug('Sequence legnth is larger than model_max_length: '+str(len(sentence))+'\t'+str(len(sentence)//block_size+1))
            for i in range(0, len(sentence), block_size):
                end = min(i+block_size, len(sentence))
                #logging.debug('\t Adding substring: '+str(i)+' - '+str(end))
                examples.append(tokenizer.encode(sentence[i:end]))
    # Create tensors
    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])
    # Create dataset
    input_tensor = tf.ragged.constant(inputs).to_tensor()
    label_tensor = tf.ragged.constant(labels).to_tensor()
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor))
    return(dataset)


def data_generator(data_files, tokenizer, batch_size=16, buffer_size=10000):
    '''  '''
    # Shuffle data_files
    file_ordering = np.random.permutation(len(data_files))
    for file_idx in file_ordering[:3]:
        dataset = process_line_sentence_file(data_files[file_idx], tokenizer)
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)
        for example in list(dataset.as_numpy_iterator()):
            yield example[0], example[1]
    return

# Test clm function
def test_clm(model, tokenizer, starting_text='一日之計在於晨，', max_length=50, num_trials=5):
    # Parse seeding string
    input_ids = tokenizer.encode(starting_text, return_tensors='tf')
    # Generate text
    generated = model.generate(input_ids, 
                            max_length=max_length,  
                            num_return_sequences=num_trials,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.5,
                            top_p=0.92,
                            temperature=.85,
                            do_sample=True,
                            top_k=125,
                            early_stopping=True)
    # Output
    output=[]
    for i in range(num_trials):
        text = tokenizer.decode(generated[i], skip_special_tokens= True)    # Decode the generated text
        text = text.replace(' ','')                                         # Remove spaces between tokens
        trial = {'id':i+1, 'text': text}
        print(text+'\n')
        output.append(trial)
    return(output)

def train_model_by_files(data_files, model, tokenizer, epochs=3, batch_size=16, buffer_size=10000):
    '''  '''
    # Prepare to monitor the training
    history_loss = []
    history_gen =[]
    # Shuffle data_files
    file_ordering = np.random.permutation(len(data_files))
    for file_idx in file_ordering[:3]:
        dataset = process_line_sentence_file(data_files[file_idx], tokenizer)
        # Create data batches
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=False)
        history = model.fit(dataset, epochs=epochs, batch_size=batch_size, steps_per_epoch=len(dataset))
        history_loss.append(history)
        generated = test_clm(model, tokenizer)
        history_gen.append(generated)
    return((history_loss, history_gen, model))

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
    parser.add_argument('--newmodel', action='store_true')
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
    if args.newmodel:
        model = initialize_gpt2()
    else:
        model = initialize_gpt2(args.model_path)
    tokenizer = BertTokenizerFast.from_pretrained('../model/tokenizer/')
    # 3. Prepare data
    dg = data_generator(data_files, tokenizer, batch_size=int(args.batch_size))
    # 4. Train model
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    history, generated, model = train_model_by_files(data_files, model, tokenizer, epochs=EPOCHS, batch_size=BATCH_SIZE)
    #test_clm(model, tokenizer)
    # 5. Save
    model.save_pretrained(args.model_path)
    tokenizer.save_pretrained(args.model_path)
    with open(args.model_path+'/history.log', 'w') as f:
        f.write(history)
    with open(args.model_path+'/generated.log', 'w') as f:
        f.write(generated)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()
