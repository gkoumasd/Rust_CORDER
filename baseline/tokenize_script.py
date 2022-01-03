#import sys
# To import upper level modules
#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))
import argparse
from tokenizer import Tokenizer


def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default="data")
    parser.add_argument('--token_voc',default="vocab/voc.txt")
    opt = parser.parse_args()
    return opt


def main(opt):
    data_path = opt.data_path
    token_voc = opt.token_voc
    
    tokenizer = Tokenizer(data_path, token_voc)
    tokenizer.tokenization()
    
if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)    
    
