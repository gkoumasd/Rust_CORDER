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
from data_loader import Load_Data
from sklearn.metrics import accuracy_score

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def train(opt, token_voc = 'vocab/voc.txt'):
    
    voc_size = 0
    dict_file = open(token_voc)
    for index, token in enumerate(dict_file):
        voc_size = index+1
        
    model = CNNClassifier(voc_size)    
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1])
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(opt.cuda))
        model.cuda(int(opt.cuda))
        loss_function = nn.BCELoss().cuda(int(opt.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        model.to(device)
        loss_function = nn.BCELoss()    
        
        
    #Load Data
    train_dataloader = Load_Data(opt.train_data,opt.batch_size)
    train_data = train_dataloader.loader()  
    
    val_dataloader = Load_Data(opt.val_data,opt.batch_size)
    val_data = val_dataloader.loader()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) * opt.epochs
    
    training_stats = []
    best_loss = 10000
    trigger_times = 0
    
    # For each epoch...
    for epoch_i in range(0, opt.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epochs))
        
        print('Training...')
        model.train()
        
        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_train_acc = 0
        n_labels = []
        n_predict = []
        
        for step, data in tqdm(enumerate(train_data)):
            
            #Load data
            if torch.cuda.device_count() > 1:
                input_ids = data[0].cuda(non_blocking=True)
                #attention_masks  = data[1].cuda(non_blocking=True)
                labels = data[2].cuda(non_blocking=True)
            else:    
                input_ids = data[0].to(device)
                #attention_masks  = data[1].to(device)
                labels = data[2].to(device)
                
                
                
            # Zero gradients
            optimizer.zero_grad()
            
            #Calculate loss
            outputs = model(input_ids)
            loss = loss_function(outputs.squeeze(dim=-1), labels.float())
            total_train_loss += loss.item()  
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # PREDICTIONS 
            pred = np.round(outputs.detach())
            labels = np.round(labels.float().detach())             
            n_predict.extend(pred.tolist())
            n_labels.extend(labels.tolist())
            total_train_acc += accuracy_score(n_labels,n_predict)
            
            if step%opt.statistic_step==0 and step!=0:
                loss_step = total_train_loss/step
                acc_step = total_train_acc/step
                print('Train loss per %d steps:%0.3f'%(step,loss_step))
                print('Train acc per %d steps:%0.3f'%(step,acc_step))
                print(classification_report(n_labels, n_predict, target_names=['Safe','Unsafe'], zero_division=0))
            
        print("")
        print('Training resuts')        
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data)
        avg_train_acc = total_train_acc/ len(train_data)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average training acc: {0:.2f}".format(avg_train_acc))
        print(classification_report(n_labels, n_predict, target_names=['Save','Unsafe'], zero_division=0))     


        print("")
        print("Running Validation...")
            
        model.eval()
        
        n_predict = []
        n_labels = []
        total_eval_loss = 0
        total_eval_acc = 0
        
        # Evaluate data for one epoch
        for step, data in tqdm(enumerate(val_data)):
            #Load data
            if torch.cuda.device_count() > 1:
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
                loss = loss_function(outputs.squeeze(dim=-1), labels.float())
                
                # PREDICTIONS 
                pred = np.round(outputs.detach())
                labels = np.round(labels.float().detach())             
                n_predict.extend(pred.tolist())
                n_labels.extend(labels.tolist())
                total_eval_acc += accuracy_score(n_labels,n_predict)
                
                
                if step%opt.statistic_step==0 and step!=0:
                    loss_step = total_eval_loss/step
                    acc_ste = total_eval_acc/step
                    print('Val loss per %d steps:%0.3f'%(step,loss_step))
                    print('Val acc per %d steps:%0.3f'%(step,acc_step))
                    print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))
        
        print("")
        print('Val resuts')        
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_data)
        avg_val_acc = total_eval_acc / len(val_data)
        print("  Average val loss: {0:.2f}".format(avg_val_loss))
        print("  Average val acc: {0:.2f}".format(avg_val_acc))
        print(classification_report(n_labels, n_predict, target_names=['Non-sequential','Sequential'], zero_division=0))        
        
        # Record all statistics from this epoch.
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Acc': avg_train_acc,
            'Valid. Acc': avg_val_acc
            })
        
        if avg_val_loss<best_loss:
            trigger_times = 0
            best_loss = avg_val_loss
            print('Found better model')
            print("Saving model to %s"%opt.model_path)
            model_to_save = model
            torch.save(model_to_save, opt.model_path)
        else:
            trigger_times += 1
            if trigger_times>= opt.patience:
                print('Early Stopping!')
                break
            
        #Save statistics to a csv file
        df = pd.DataFrame.from_dict(training_stats)
        df.to_csv(opt.model_path.split('.')[0] + '.csv', sep=';', encoding='utf-8')
        
    
    print("")
    print("Training complete!")     

        
if __name__ == "__main__":
    opt = parse_arguments()
    print(opt)
    USE_CUDA = torch.cuda.is_available()
