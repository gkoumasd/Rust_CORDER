from tqdm import *
import os
from transformers import RobertaTokenizer
import torch
import pickle
from torch.utils.data import TensorDataset


class Tokenizer():
     def __init__(self, data_path: str):
         self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
         self.data_path = data_path
         
     def tokenize(self):
         
          # Tokenize all of the code snippets and map the tokens to thier tokens IDs.
          input_ids = []
          attention_masks = []
          token_type_ids = []
          lst_files = []
          for subdir , dirs, files in os.walk(self.data_path):
              for file in tqdm(files):
                  file_path = os.path.join(subdir, file)
                  if file.endswith(".rs"):
                      file_name = file_path.replace('\\','/').split("/")[1]
                
                      lst_files.append(file_name)
                      
                      with open(file_path, "r", errors='ignore') as f:
                            code_snippet = str(f.read())
                      
                            
                        
                      nl = ''
                      code = ''
                      for line in code_snippet.split('\n'):
                          if line.startswith('#'): #it's directive
                              nl +=line
                              nl +=' '
                          else: #it#s code   
                               code += line
                       
                      
                            
                      encoded_dict = self.tokenizer.encode_plus(
                        code,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 512,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_token_type_ids= True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                       )    
                      
                      
                      
                      
                      # Add the encoded code snippet to the list.    
                      input_ids.append(encoded_dict['input_ids'])
                      
                      # And its attention mask (simply differentiates padding from non-padding).
                      attention_masks.append(encoded_dict['attention_mask'])
                      
                      token_type_ids.append(encoded_dict['token_type_ids'])
                      
                      
                      
          # Convert the lists into tensors.
          input_ids = torch.cat(input_ids, dim=0)
          attention_masks = torch.cat(attention_masks, dim=0)
          token_type_ids = torch.cat(token_type_ids, dim=0)
          
          # Combine the training inputs into a TensorDataset.
          dataset = TensorDataset(input_ids, attention_masks, token_type_ids)
          
          #Save to pickle files
          self.save2pkl('features', dataset)
          self.save2pkl('files', files)
         
          
          
     def save2pkl(self, flag, data):
        self.flag = flag
        self.data = data

        with open('data/'+ self.flag  + '.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)                   