import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from codeBERT.load_data import Load_Data
from codeBERT.argument_parser import parse_arguments
from codeBERT.model import RobertaClass
from transformers import get_linear_schedule_with_warmup


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def train(opt):
    
    #Load Data
    train_dataloader = Load_Data('train',opt.batch_size)
    train_data = train_dataloader.loader()
    
    val_dataloader = Load_Data('val',opt.batch_size)
    val_data = val_dataloader.loader()
    
    #Load the model
    model = RobertaClass()
    model.to(device)
    print('CodeBERT has {:} different named parameters.\n'.format(len(list(model.named_parameters()))))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_data) * opt.epochs
    # Create the learning rate scheduler.
    #However scheduler is not needed for fine-tuning
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps = 0, # Default value in run_glue.py
                num_training_steps = total_steps)
    
    loss_function = nn.CrossEntropyLoss()
    
    # We'll store a number of quantities such as training and validation loss etc
    training_stats = []
    
    #Path to save the best model
    output_dir = 'best/'
    best_loss = 10000
    # For each epoch...
    for epoch_i in range(0, opt.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epochs))
        print('Training...')
        
        model.train()
        
        # Reset the total loss for this epoch.
        total_train_loss = 0
        n_predict = []
        n_labels = []
        
        for step, data in tqdm(enumerate(train_data)):
            
            #Load data
            input_ids = data[0].to(device)
            attention_masks  = data[1].to(device)
            token_type_ids = data[2].to(device)
            labels = data[3].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            #Calculate loss
            outputs = model(input_ids, attention_masks, token_type_ids)
            loss = loss_function(outputs, labels)
            total_train_loss += loss.item()
            
            #Precision, Recall, F1
            max_val, max_idx = torch.max(outputs.data, dim=1)
            n_predict.extend(max_idx.tolist())
            n_labels.extend(labels.tolist())
            
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
            
            if step%opt.statistic_step==0:
                 loss_step = total_train_loss/step
                 train_accuracy = accuracy_score(n_labels, n_predict).item()
                 train_precision = precision_score(n_labels, n_predict, average='micro').item()
                 train_recall= recall_score(n_labels, n_predict, average='micro').item()
                 train_f1 = f1_score(n_labels, n_predict, average='micro').item()
                 print('Train loss per %d steps:%0.3f'%(step,loss_step))
                 print('Train accuracy per %d steps:%0.1f'%(step, train_accuracy))
                 print('Train micro precision per %d steps:%0.1f'%(step,train_precision))
                 print('Train micro recall per %d steps:%0.1f'%(step,train_recall))
                 print('Train micro f1 score per %d steps:%0.1f'%(step,train_f1))
                 
                 
                 
                 
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data) 
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
            
        print("")
        print("Running Validation...")
        
        model.eval()
        
        n_predict = []
        n_labels = []
        total_eval_loss = 0
        # Evaluate data for one epoch
        for step, data in tqdm(enumerate(val_data)):
            #Load data
            input_ids = data[0].to(device)
            attention_masks  = data[1].to(device)
            token_type_ids = data[2].to(device)
            labels = data[3].to(device)
            
            with torch.no_grad():   
                #Calculate loss
                outputs = model(input_ids, attention_masks, token_type_ids)
                loss = loss_function(outputs, labels)
                
                #Precision, Recall, F1
                max_val, max_idx = torch.max(outputs.data, dim=1)
                n_predict.extend(max_idx.tolist())
                n_labels.extend(labels.tolist())
                
                # Accumulate the validation loss.
                total_eval_loss += loss.item()  
                
                if step%opt.statistic_step==0:
                    loss_step = total_eval_loss/step
                    val_accuracy = accuracy_score(n_labels, n_predict).item()
                    val_precision = precision_score(n_labels, n_predict, average='micro').item()
                    val_recall= recall_score(n_labels, n_predict, average='micro').item()
                    val_f1 = f1_score(n_labels, n_predict, average='micro').item()
                 
                    print('Train loss per %d steps:%0.3f'%(step,loss_step))
                    print('Train accuracy per %d steps:%0.1f'%(step,val_accuracy))
                    print('Train micro precision per %d steps:%0.1f'%(step,val_precision))
                    print('Train micro recall per %d steps:%0.1f'%(step,val_recall))
                    print('Train micro f1 score per %d steps:%0.1f'%(step,val_f1))
        
        # Report the final accuracy for this validation run.
        print("")
        print('Validation resuts')
        print('Train accuracy per %d steps:%0.1f'%(step,accuracy_score(n_labels, n_predict).item()))
        print('Train micro precision per %d steps:%0.1f'%(step,precision_score(n_labels, n_predict, average='micro').item()))
        print('Train micro recall per %d steps:%0.1f'%(step,recall_score(n_labels, n_predict, average='micro').item()))
        print('Train micro f1 score per %d steps:%0.1f'%(step,f1_score(n_labels, n_predict, average='micro').item()))
                 
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_data) 
        print("")
        print("  Average validation loss: {0:.2f}".format(avg_val_loss))     
        
        
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Accuracy.': train_accuracy,
                'Valid. Accuracy': val_accuracy,
                'Training Precision': train_precision,
                'Valid. Precision':val_precision,
                'Training Recall':train_recall,
                'Valid. Recall':val_recall,
                'Training f1':train_f1,
                'Valid. f1':val_f1,
            }
        )
        
        if avg_val_loss<best_loss:
            print('Found better model')
            print("Saving model to %s" % output_dir)
            model_to_save = model
            torch.save(model_to_save, output_dir)
    
    
    #Save statistics to a txt file
    textfile = open("best/training_stats.txt", "w")
    for element in training_stats:
        textfile.write(str(element) + "\n")
    textfile.close()
    
    print("")
    print("Training complete!")
               
        

if __name__ == "__main__":
    opt = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    train(opt)
