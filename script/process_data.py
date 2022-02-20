import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
import argparse

from util.data.data_processor.pycparser_data_processor import PycParserDataProcessor
from util.data.data_processor.treesitter_c_data_processor import TreeSitterCDataProcessor
from util.data.data_processor.srcml_data_processor import SrcmlDataProcessor
from util.data.data_processor.treesitter_rs_data_processor import TreeSitterRustDataProcessor
from util.data.data_processor.treesitter_asm_data_processor import TreeSitterASMDataProcessor



def parse_arguments(): 
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_path', default="../OJ_pycparser_train_test_val/val")
    #parser.add_argument('--node_type_vocab_path',default="../vocab/pycparser/node_type/type.txt")
    #parser.add_argument('--node_token_vocab_path', default="../vocab/pycparser/node_token/token.txt")
    #parser.add_argument('--parser', type=str, default="pycparser", help="pycparser, treesitterc, srcml")
    
    parser.add_argument('--data_path', default="treesitter_rs_train_test_val/train")
    parser.add_argument('--node_type_vocab_path',default="vocab/treesitter_rs/node_type/type.txt")
    parser.add_argument('--node_token_vocab_path', default="vocab/treesitter_rs/node_token/token.txt")
    parser.add_argument('--parser', type=str, default="treesitter_rs", help="pycparser, treesitterc, srcml, treesitter_asm")
#    parser.add_argument('--data_path', default="treesitter_asm_train_test_val/train")
#    parser.add_argument('--node_type_vocab_path',default="vocab/treesitter_asm/node_type/type.txt")
#    parser.add_argument('--node_token_vocab_path', default="vocab/treesitter_asm/node_token/token.txt")
#    parser.add_argument('--parser', type=str, default="treesitter_asm", help="pycparser, treesitterc, srcml, treesitter_rs, treesitter_asm")
    opt = parser.parse_args()
    return opt

def main(opt):
    data_path = opt.data_path
    node_type_vocab_path = opt.node_type_vocab_path
    node_token_vocab_path = opt.node_token_vocab_path
    parser = opt.parser
    
    if parser == "pycparser":
        processor = PycParserDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)
    if parser == "treesitterc":
        processor = TreeSitterCDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)
    if parser == "srcml":
        processor = SrcmlDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)
    if parser == "treesitter_rs":
        processor = TreeSitterRustDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)
    if parser == "treesitter_asm":
        processor = TreeSitterASMDataProcessor(node_type_vocab_path, node_token_vocab_path, data_path, parser)

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
   
