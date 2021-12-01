import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
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
    
   
    # For each epoch...
    for epoch_i in range(0, opt.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, opt.epochs))
        print('Training...')
        
        model.train()
        
        # Reset the total loss for this epoch.
        total_train_loss = 0
        
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
                 print('Train loss per %d steps:0.3f'%(step,loss_step))
                 
            
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data) 
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
            

if __name__ == "__main__":
    opt = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    train(opt)
