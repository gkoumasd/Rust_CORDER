from tqdm import *
import os
import numpy as np
import re


#token_voc = vocab/voc.txt
class Tokenizer():
    def __init__(self, data_path: str, token_voc = 'vocab/voc.txt'):
        self.data_path = data_path
        self.token_voc = token_voc
        self.dict = []
        self.dict.append('<UKN>')
        
    def tokenization(self):
        if os.path.exists(self.token_voc):
            os.remove(self.token_voc)
            
        for subdir , dirs, files in os.walk(self.data_path): 
             for file in tqdm(files):
                 file_path = os.path.join(subdir, file)
                 if file.endswith(".rs"):
                     #print(file)
                     
                     with open(file_path, "r", errors='ignore') as f:
                         code_snippet = str(f.read())
                     
                     extr_tokens = np.array(self.extract_tokens(code_snippet)) 
                     #keep unique tokens
                     extr_tokens =  np.unique(extr_tokens)
                     
                     #Update 
                     self.dict = np.append(self.dict, extr_tokens,0)
                     self.dict  = np.unique(self.dict)
        
        #Save tokens to txt file.
        voc_file = open("vocab/voc.txt", "w")             
        for i, row in enumerate(self.dict):
            if i < len(self.dict)-1:
                voc_file.write(row+'\n')
            else:
                voc_file.write(row)
        voc_file.close()    
            
                     
                          
                         
    def extract_tokens(self, code_snippet: str):
        
        code_snippet = code_snippet.replace('\n', '')
        
        extr_tokens = re.findall("[A-Za-z_]\w*|[!\#\$%\&\*\+:\-/<=>\?@\\\^_\|\~]",code_snippet)
        
        tokens_lst = []
        for token in extr_tokens:
            tmp_tokens = [t for t in  token.lower().split('_')]
            tokens_lst.extend(tmp_tokens)
            
        return tokens_lst    
    


        