import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import argparse
from tokenizer import Tokenizer


def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default="data_train_test_val/train")
    parser.add_argument('--flag',default="train")
    parser.add_argument('--file_type',default="asm")
    opt = parser.parse_args()
    return opt


def main(opt):
    data_path = opt.data_path
    flag = opt.flag
    file_type = opt.file_type
    
    tokenizer = Tokenizer(data_path, flag, file_type)
    tokenizer.tokenize()
    
if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)    
    