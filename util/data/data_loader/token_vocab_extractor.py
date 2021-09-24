from tqdm import *
import os
import numpy as np
from functools import lru_cache
from typing import List
import sys
import re

REGEX_TEXT = ("(?<=[a-z0-9])(?=[A-Z])|"
              "(?<=[A-Z0-9])(?=[A-Z][a-z])|"
              "(?<=[0-9])(?=[a-zA-Z])|"
              "(?<=[A-Za-z])(?=[0-9])|"
              "(?<=[@$.'\"])(?=[a-zA-Z0-9])|"
              "(?<=[a-zA-Z0-9])(?=[@$.'\"])|"
              "_|\\s+")

if sys.version_info >= (3, 7):
    import re
    SPLIT_REGEX = re.compile(REGEX_TEXT)
else:
    import regex
    SPLIT_REGEX = regex.compile("(?V1)"+REGEX_TEXT)


class TokenVocabExtractor():
    def __init__(self, input_data_path: str, node_token_vocab_path:str):
        self.input_data_path = input_data_path
        self.node_token_vocab_path = node_token_vocab_path
      
        
    def create_vocab_from_dir(self): 
      
       
        #Extract tokens fron code snippets
        all_tokens = []        
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                #print(file_path)
                with open(file_path, "r", errors='ignore') as f:
                    data = str(f.read())
                    tokens = self.split_identifier_into_parts(data)
                    
                for token in tokens:
                    if (len(token)>1):
                        token = re.sub('[\(\)\[\]]', '', token)
                    all_tokens.append(token)
                    
        unique_tokens = np.array(all_tokens)  
        unique_tokens = np.unique(unique_tokens).tolist() 
        #unique_tokens = unique_tokens[:64600]
        unique_tokens = unique_tokens[:60000]
                 
        if os.path.exists(self.node_token_vocab_path):
            os.remove(self.node_token_vocab_path)
        
        token_file = open(self.node_token_vocab_path, "w")
        for i, token in enumerate(unique_tokens):
            if (len(token)==0):
                continue
            if (i<len(unique_tokens)-1):
                token_file.write(token + "\n")
            else:
                token_file.write(token)
        token_file.close() 
        print('Token file has been created!')
        
        
        
    def split_identifier_into_parts(self,identifier: str) -> List[str]:
        """
        Split a single identifier into parts on snake_case and camelCase
        """
        identifier_parts = list(s.lower() for s in SPLIT_REGEX.split(identifier) if len(s)>0)

        if len(identifier_parts) == 0:
            return [identifier]
        return identifier_parts   

                            
                         
                        
    


     
