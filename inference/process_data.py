import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
import argparse
from treesitter_rust_data_processor import TreeSitterRustDataProcessor
def parse_arguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="data/treesitter_rust-buckets.pkl")
    parser.add_argument('--node_type_vocab_path',default="../vocab/treesitter_rust/node_type/type.txt")
    parser.add_argument('--node_token_vocab_path', default="../vocab/treesitter_rust/node_token/token.txt")
    parser.add_argument('--parser', type=str, default="treesitter_rust", help="pycparser, treesitterc, srcml, treesitter_rust")
    parser.add_argument('files', nargs='+', help='file to infer', type=open)
    opt = parser.parse_args()
    return opt

def main(opt):
    node_type_vocab_path = opt.node_type_vocab_path
    node_token_vocab_path = opt.node_token_vocab_path
    parser = opt.parser
    processor = TreeSitterRustDataProcessor(node_type_vocab_path, node_token_vocab_path, opt.files, opt.data_path, parser)

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
   

