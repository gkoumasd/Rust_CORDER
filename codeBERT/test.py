import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from load_data import Load_Data
from argument_parser import parse_arguments





def test(opt):
    #Load Data
    test_dataloader = Load_Data('test',opt.batch_size)
    test_data = test_dataloader.loader()
    
    #Load the model
    model = torch.load(os.path.join(opt.model_path,'codeBERT_asm32_pl.bin'))
    model.to(device)
    
    print("")
    print("Running Testing...")
        
    model.eval()
    
    loss_function = nn.CrossEntropyLoss()
    
    n_predict = []
    n_labels = []
    total_test_loss = 0
    for step, data in tqdm(enumerate(test_data)):
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
            total_test_loss += loss.item() 
            
            
            if step%opt.statistic_step==0 and step!=0:
                loss_step = total_test_loss/step
                val_accuracy = accuracy_score(n_labels, n_predict).item()
                val_precision = precision_score(n_labels, n_predict, average='micro').item()
                val_recall= recall_score(n_labels, n_predict, average='micro').item()
                val_f1 = f1_score(n_labels, n_predict, average='micro').item()
                 
                print('Test loss per %d steps:%0.3f'%(step,loss_step))
                print(classification_report(n_labels, n_predict, target_names=['safe','unsafe']))
                 
    # Report the final accuracy for this validation run.
    print("")
    print('Test resuts')
    print(classification_report(n_labels, n_predict, target_names=['safe','unsafe']))
    
    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_data) 
    print("")
    print("  Average test loss: {0:.2f}".format(avg_test_loss))  


if __name__ == "__main__":
    opt = parse_arguments()
    print(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    test(opt)

