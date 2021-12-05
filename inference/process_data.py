import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import argparse

from treesitter_rust_data_processor import TreeSitterRustDataProcessor



def parse_arguments(): 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', default="inference/data/")
    parser.add_argument('--node_type_vocab_path',default="vocab/treesitter_rust/node_type/type.txt")
    parser.add_argument('--node_token_vocab_path', default="vocab/treesitter_rust/node_token/token.txt")
    parser.add_argument('--parser', type=str, default="treesitter_rust", help="pycparser, treesitterc, srcml, treesitter_rust")
    opt = parser.parse_args()
    return opt

def main(opt):
    data_path = opt.data_path
    node_type_vocab_path = opt.node_type_vocab_path
    node_token_vocab_path = opt.node_token_vocab_path
    parser = opt.parser
    
    processor = TreeSitterRustDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
   

