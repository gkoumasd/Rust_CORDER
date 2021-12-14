#import sys
#from pathlib import Path
# To import upper level modules
#sys.path.append(str(Path('.').absolute().parent))
import argparse
from tokenizer import Tokenizer


def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default="data/")
    opt = parser.parse_args()
    return opt


def main(opt):
    data_path = opt.data_path
    
    tokenizer = Tokenizer(data_path)
    tokenizer.tokenize()
    
if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)    
    