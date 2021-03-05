# -*- coding: utf-8 -*-
from __future__ import print_function
import logging, os, argparse
from transformers import BertTokenizerFast, AutoModelForCausalLM
import pandas as pd

#-----------------------------------------------------------------------
def main():
    '''    '''
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='Retrieve DBZ data for further processing.')
    parser.add_argument('--input', '-i', default='', help='the seeding string.')
    parser.add_argument('--output', '-o', default='generated.csv', help='the output txt file.')
    parser.add_argument('--length', default=30, type=int, help='the maximal length of generated text.')
    parser.add_argument('--trials', default=5, type=int, help='the number of trials.')
    parser.add_argument('--logfile', '-l', default=None, help='the log file.')
    args = parser.parse_args()
    # Set up logging
    if not args.logfile is None:
        logging.basicConfig(level=logging.DEBUG, filename=args.logfile, filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG)
    # Prompt the setting
    logging.info('Generating text with pre-trained GPT-2.\n  Seed text: ' + args.input
        +'\n  Max length: ' + str(args.length)
        +'\n  Trial: ' + str(args.trials))
    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese', clean_text=True)
    model = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese')
    # Parse seeding string
    input_ids = tokenizer.encode(args.input, return_tensors='pt')
    # Generate text
    generated = model.generate(input_ids, 
                            max_length=args.length,  
                            num_return_sequences=args.trials,
                            no_repeat_ngram_size=2,
                            repetition_penalty=1.5,
                            top_p=0.92,
                            temperature=.85,
                            do_sample=True,
                            top_k=125,
                            early_stopping=True)
    # Output
    output=[]
    for i in range(args.trials):
        text = tokenizer.decode(generated[i], skip_special_tokens= True)    # Decode the generated text
        text = text.replace(' ','')                                         # Remove spaces between tokens
        trial = {'id':i+1, 'text': text}
        print(text)
        output.append(trial)
    pd.DataFrame(output).to_csv(args.output, index=False)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()