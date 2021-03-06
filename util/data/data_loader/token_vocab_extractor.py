from tqdm import *
import os
import numpy as np
from functools import lru_cache
from typing import List
import sys
import re
from nltk.stem import PorterStemmer



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
    def __init__(self, data_path: str, node_token_vocab_path:str, language:str):
        self.data_path = data_path
        self.language = language
        self.node_token_vocab_path = node_token_vocab_path
        self.stemer = PorterStemmer()
        
    def create_vocab_from_dir(self): 
        #Please don't remove the statement.
        #Extraction of tokens should be executed 
        if os.path.exists(self.node_token_vocab_path):
            os.remove(self.node_token_vocab_path)
        
        if not os.path.exists(self.node_token_vocab_path):
            #Extract tokens fron code snippets
            all_tokens = []
            for subdir , dirs, files in os.walk(self.data_path): 
                for file in tqdm(files):
                    file_path = os.path.join(subdir, file)
                    
                    if file.endswith('.' + self.language):
                       
                        stem_tokens = []
                        
                        with open(file_path, "r", errors='ignore') as f:
                            code_snippet = str(f.read())   
                            
                            
                        nl = ''
                        code = ''
                        if self.language=='rs': #works on high-level language
                            for line in code_snippet.split('\n'):
                                if line.startswith('#') or line.startswith('//'): #it's directive or comment
                                    nl +=line
                                    nl +=' '
                                else: #it#s code   
                                     code += line
                        elif self.language=='asm': #works on low-level language 
                            #this version disrecard language information, i.e., comments
                             for line in code_snippet.split('\n'):
                                 line = line.split('#')
                                 if len(line[0])>0 and '#' not in line[0]:
                                     code += line[0] + ' '
                                 elif len(line)>1 and len(line[1])>0 and '#' not in line[1]: #When the comment followed by code
                                     code += line[1] + ' '    
                            
                            #Remove non alpharithmetic characters from code snippet
                            #data = re.sub(r'\W+', ' ', data)
        
                        tokens = self.split_identifier_into_parts(code) 
                        tokens.sort()
                        #Apply stem so as to reduce noise in voc
                        for token in tokens:
                            #print(token, '->',self.stemer.stem(token) )
                            stem_tokens.append(self.stemer.stem(token))
                        all_tokens.extend(stem_tokens)    
                    
                    
            unique_tokens = np.unique(all_tokens).tolist()
            ## Add the UKN token in the beggining
            unique_tokens.sort()
            unique_tokens.insert(0,'<UKN>')
           
            #unique_tokens = unique_tokens[:55000]
            token_file = open(self.node_token_vocab_path, "w")
            for i, token in enumerate(unique_tokens):
                if (len(token)==0):
                    continue
                if (i<len(unique_tokens)-1):
                    token_file.write(token + "\n")
                else:
                    token_file.write(token)
            token_file.close()
            print('Token file has been created!\n')
        print("Token file %s\n" % self.node_token_vocab_path)
        
    def split_identifier_into_parts(self,identifier: str) -> List[str]:
        """
        Split a single identifier into parts on snake_case and camelCase
        """
        identifier_parts = list(s.lower() for s in SPLIT_REGEX.split(identifier) if len(s)>0)

        if len(identifier_parts) == 0:
            return [identifier]
        return identifier_parts