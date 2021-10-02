from util.data.data_processor.base_data_processor import DataProcessor
import os
from tqdm import trange
from tqdm import *
# from util.data.data_processor.ast_parser import ASTParser
from tree_sitter_parsers import ASTParser
from util import identifier_splitting
from util.data.data_loader.token_vocab_extractor import TokenVocabExtractor
from typing import List, Dict, Iterable

ast_parser = ASTParser("rust")

def process_a_job(job: int, r: (Iterable, List[int]), file_paths: Dict[int, List[str]], node_type_lookup, node_token_lookup, token_vocab):
    trees = []
    for i in r:
        file_path = file_paths[i - 1]
        file_path = file_path.replace('\\','/') #Windows version        
        #Extract the classification label.
        file_path_splits = file_path.split("/")
        if (file_path_splits[-2]=='safe'):
            label = 0
        else:
            label = 1
        with open(file_path, "rb") as f:
            code_snippet = f.read()
        #Create a AST representation
        ast = ast_parser.parse(code_snippet)
        #Simplify AST to a nested dictionary
        tree, sub_tokens, size  = simplify_ast(ast, code_snippet, node_type_lookup, node_token_lookup, token_vocab)        
        tree_data = {
                "tree": tree,
                "size": size,
                "label": label,
                "sub_tokens": sub_tokens,
                "file_path": file_path
            }
        trees.append(tree_data)
    return trees

            
def simplify_ast(tree, text, node_type_lookup, node_token_lookup, token_vocab): #tree-> ast, text->code_snippet
    
    root = tree.root_node
    #root.sexp()
    
    ignore_types = ["\n"]
    num_nodes = 0
    root_type = str(root.type)
    root_type_id = node_type_lookup.get(root_type.upper())
    queue = [root]
    
    root_json = {
        "node_type": root_type,
        "node_type_id": root_type_id,
        "node_tokens": [],
        "node_tokens_id": [],
        "children": []
    }
    
    queue_json = [root_json]
    
    tree_tokens = []
    
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        num_nodes += 1
        
        for child in current_node.children:
            child_type = str(child.type)
            
            
            if child_type not in ignore_types:
                queue.append(child)
                child_type_id = node_type_lookup.get(child_type.upper())
                #print(child_type_id, child_type.upper() )
                
                child_token = ""
                child_sub_tokens_id = []
                child_sub_tokens = []
                
                has_child = len(child.children) > 0
                
                
                if not has_child:
                    child_token = text[child.start_byte:child.end_byte]
                    child_sub_tokens  = token_vocab.split_identifier_into_parts(child_token.decode('utf-8'))
                    child_sub_tokens_id = [node_token_lookup.get(child_sub_token) for child_sub_token in child_sub_tokens]
                    #Replace None values with UKN_token indexed to 0
                    child_sub_tokens_id = [0 if child_sub_token_id is None else child_sub_token_id for child_sub_token_id in child_sub_tokens_id]
                    #print(child_sub_tokens_id, child_sub_tokens)
                    #child_sub_tokens = subtokens.split(' ')
                    #subtokens = " ".join(identifier_splitting.split_identifier_into_parts(child_token.decode('utf-8')))
                    #child_sub_tokens = self.token_vocab.tokenize(subtokens)
                    
                
                if len(child_sub_tokens_id) == 0:
                    child_sub_tokens_id.append(0)
                else:
                    child_sub_tokens_id = [x for x in child_sub_tokens_id if x != 0]    
                
                
                child_json = {
                    "node_type": child_type,
                    "node_type_id": child_type_id,
                    "node_tokens": child_sub_tokens,
                    "node_tokens_id": child_sub_tokens_id,
                    "children": []
                }
                
                tree_tokens.extend(child_sub_tokens)
                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
            
    tree_tokens = list(set(tree_tokens))        
    return root_json, tree_tokens, num_nodes        

class TreeSitterRustDataProcessor(DataProcessor):
    def __init__(self, node_type_vocab_path, node_token_vocab_path, data_path, parser):
        self.token_vocab = TokenVocabExtractor(data_path,node_token_vocab_path)
        super().__init__(node_type_vocab_path, node_token_vocab_path, data_path, parser)

    def load_program_data(self, directory):
        rust_files = []
        for subdir , dirs, files in os.walk(directory): 
            for file in files:
                if file.endswith(".rs"):
                    file_path = os.path.join(subdir, file)
                    rust_files.append(file_path)                           
        import multiprocessing as mp
        n_processes = mp.cpu_count()
        pool = mp.Pool(n_processes)
        jobs = [(n, range(n, len(rust_files) + 1, n_processes), rust_files, self.node_type_lookup, self.node_token_lookup, self.token_vocab) 
            for n in range(1, n_processes + 1)]
        res = pool.starmap_async(process_a_job, iterable=jobs).get()
        pool.close()
        pool.join()
        trees = []
        for t in res:
            trees.extend(t)
        print("Total processed files : %d\n" % len(trees))
        return trees
