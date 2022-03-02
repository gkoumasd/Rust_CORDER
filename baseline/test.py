import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
from argument_parser import parse_arguments
import pickle
from model.cnn import CNNClassifier
from model.bilstm import BiLSTMClassifier
from data_loader import Load_Data
from sklearn.metrics import accuracy_score

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def test(opt, token_voc = 'vocab/voc.txt'):
    
    voc_size = 0
    dict_file = open(token_voc)
    for index, token in enumerate(dict_file):
        voc_size = index+1
        
        
    if opt.model == 'CNN':    
        model = CNNClassifier(voc_size)
    elif opt.model == 'BiLSTM': 
        model = BiLSTMClassifier(voc_size)
    
    if torch.cuda.device_count() > 1 and opt.model == 'CNN':
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1])
        if os.path.exists(opt.model_path):
            model = torch.load(opt.model_path)
            print('Pretrained model has been loaded')
        else:
            print('Pretrained model does not exist!!!')
            
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(opt.cuda))
        model.cuda(int(opt.cuda))
        
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(opt.model_path):
            model = torch.load(opt.model_path,map_location =device)
            print('Pretrained model has been loaded')
        else:
            print('Pretrained model does not exist!!!') 
        model.to(device)
        
        
    #Load Data
    test_dataloader = Load_Data(opt.test_data,opt.batch_size)
    test_data = test_dataloader.loader()  
    
    print("")
    print("Running Testing...")
        
    model.eval()
    
    n_predict = []
    n_labels = []
    total_eval_acc = 0
    
    for step, data in tqdm(enumerate(test_data)):
        
        #Load data
        if torch.cuda.device_count() > 1 and opt.model == 'CNN':
            input_ids = data[0].cuda(non_blocking=True)
            #attention_masks  = data[1].cuda(non_blocking=True)
            labels = data[2].cuda(non_blocking=True)
        else:    
            input_ids = data[0].to(device)
            #attention_masks  = data[1].to(device)
            labels = data[2].to(device)
            
            
        with torch.no_grad():   
            #Calculate loss
            outputs = model(input_ids)
               
            
            # PREDICTIONS 
            pred = torch.round(outputs.squeeze(dim=-1)).int()
            n_predict.extend(pred.tolist())
            n_labels.extend(labels.tolist())
            total_eval_acc += accuracy_score(n_labels,n_predict)
            
            if step%opt.statistic_step==0 and step!=0:
                acc_step = total_eval_acc/step
                print('Test acc per %d steps:%0.3f'%(step,acc_step))
                print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe'], zero_division=0))
                
                
    print("")
    print('Test resuts')        
     # Calculate the average loss over all of the batches.
    avg_val_acc = total_eval_acc / len(test_data)
    print("  Average test acc: {0:.2f}".format(avg_val_acc))
    print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe'], zero_division=0))            

if __name__ == "__main__":
    opt = parse_arguments()
    print(opt)
    USE_CUDA = torch.cuda.is_available()
    
    test(opt)
