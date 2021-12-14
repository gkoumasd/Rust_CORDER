import sys
sys.path.append('.')

import os
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import pickle5 as pickle
from load_data import Load_Data
from model import RobertaClass


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def inference():
    
    #Load Data
    infer_dataloader = Load_Data('features',1)
    infer_data = infer_dataloader.loader()
    
    
    with open('data/files.pickle', "rb") as fh:
             files = pickle.load(fh)
    
    
    #Load the model
    model = RobertaClass()
    if os.path.exists('model/codeBERT_pl.bin'):
        print('Loading the pre-trained model')
        model = torch.load('model/codeBERT_pl.bin',map_location =device)
    else:
        print('Caution! The pre-trained load model does not exist, you cannot reprocude the resutls')
    
    model.to(device)
    
    print("")
    print("Running inference...")
        
    model.eval()
        
    #Inference  
    for step, data in tqdm(enumerate(infer_data)):
        #Load data
        input_ids = data[0].to(device)
        attention_masks  = data[1].to(device)
        token_type_ids = data[2].to(device)
        file = files[step]
        
            
        with torch.no_grad():   
            #Calculate loss
            outputs = model(input_ids, attention_masks, token_type_ids)
            probabilities = F.softmax(outputs, dim=-1)
            max_val, max_idx = torch.max(outputs.data, dim=1)   
            
            if max_idx==0:
                result = 'safe'
            else:
                result = 'unsafe'
                
            print('\n %s classified as %s (prob=%.2f)'%(file, result, probabilities[0][max_idx].item()))    
            
            
            
    
    print("")
    print("Inference complete!")
               
        

if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    inference()
