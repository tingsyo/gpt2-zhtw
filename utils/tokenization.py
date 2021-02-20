# -*- coding: utf-8 -*-
''' This script tokenize the specified corpus for GPT2 model. '''

import os, argparse, logging
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

class BPE_token(object):
    def __init__(self):
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.normalizer = Sequence([
            NFKC()
        ])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def bpe_train(self, paths):
        trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>"
        ])
        self.tokenizer.train(trainer, paths)

    def save_tokenizer(self, location, prefix=None):
        if not os.path.exists(location):
            os.makedirs(location)
        self.tokenizer.model.save(location, prefix)


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
    # 
    # the folder 'text' contains all the files
    paths = [str(x) for x in Path(args.input).glob("**/*.txt")]
    tokenizer = BPE_token()
    # train the tokenizer model
    tokenizer.bpe_train(paths)
    # saving the tokenized data in our specified folder 
    save_path = 'tokenized_data'
    tokenizer.save_tokenizer(save_path)
    # done
    return(0)

#==========
# Script
#==========
if __name__ == "__main__":
    main()