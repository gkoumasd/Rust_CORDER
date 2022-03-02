from tqdm import *
import os
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset
import pickle


#token_voc = vocab/voc.txt
class FeatureExtraction():
    def __init__(self, data_path: str, max_seq: int, token_voc = 'vocab/voc.txt'):
        self.data_path = data_path
        self.max_seq = int(max_seq)
        self.token_voc = token_voc
        self.dict = {}
        
        self.dict_file = open(self.token_voc)
        for index, token in enumerate(self.dict_file):
            self.dict[token.replace('\n','')] = index+1
            
        
    def extract(self):    
        labels = []
        input_ids = []
        masks = []
        for subdir , dirs, files in os.walk(self.data_path):
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                if file.endswith(".rs") or file.endswith(".asm"):
                    #print(file)
                    
                    with open(file_path, "r", errors='ignore') as f:
                          code_snippet = str(f.read())
                          
                    label = file_path.replace('\\','/').split("/")[4]
                    #print(label)
                    if label == 'safe':
                        label = 0
                    else:
                        label = 1
                    
                    
                    extr_tokens = self.extract_tokens(code_snippet) 
                    if len(extr_tokens)>self.max_seq:
                        extr_tokens = extr_tokens[:self.max_seq]
                        
                           
                    idx_code = [self.dict.get(token) if token in self.dict else self.dict.get('<UKN>') for token in extr_tokens ]
                    mask = len(idx_code)*[1] + (self.max_seq-len(idx_code))*[0]
                    
                    #pad with 0 for seq less than max_seq length
                    idx_code = idx_code + (self.max_seq-len(idx_code))*[0]    
                    
                    
                    input_ids.append(idx_code)
                    masks.append(mask)
                    labels.append(label)
                    
                    
        # Convert the lists into tensors.
        input_ids = torch.tensor(input_ids)
        masks = torch.tensor(masks)
        labels = torch.tensor(labels)   

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, masks, labels)    
        
         
        #Save to pickle files
        self.save2pkl(self.data_path.split('/')[-1], dataset)
                    
    def save2pkl(self, flag, data):
       self.flag = flag
       self.data = data

       with open('data/'+ self.flag  + '.pickle', 'wb') as handle:
           pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)  
       print('File %s has been saved!'%('data/'+ self.flag  + '.pickle'))                
                    
    def extract_tokens(self, code_snippet: str):
         
         code_snippet = code_snippet.replace('\n', '')
         
         extr_tokens = re.findall("[A-Za-z_]\w*|[!\#\$%\&\*\+:\-/<=>\?@\\\^_\|\~]",code_snippet)
         
         tokens_lst = []
         for token in extr_tokens:
             tmp_tokens = [t for t in  token.lower().split('_')]
             tokens_lst.extend(tmp_tokens)
             
         return tokens_lst               
