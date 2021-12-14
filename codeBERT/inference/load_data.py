import os
#import pickle
import pickle5 as pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Load_Data():
     def __init__(self, flag: str, batch_size: int):
         self.flag = flag
         self.batch_size = batch_size
         self.data_path = os.path.join('data',self.flag+'.pickle')
        
    
     def loader(self):
         #Load data
         with open(self.data_path, "rb") as fh:
             dataset = pickle.load(fh)
         #dataset_file = open(self.data_path,'rb')
         #dataset = pickle.load(dataset_file)
         #dataset_file.close()
         
         #Load data on DataLoader
         if self.flag== 'train':
             # Create the DataLoaders for our training and validation sets.
             # We'll take training samples in random order. 
             dataloader = DataLoader(
                 dataset,  # The training samples.
                 sampler = RandomSampler(dataset), # Select batches randomly
                 batch_size = self.batch_size # Trains with this batch size.
             )
         else:
             # For validation and test the order doesn't matter, so we'll just read them sequentially.
             dataloader = DataLoader(
                dataset, # The validation samples.
                sampler = SequentialSampler(dataset), # Pull out batches sequentially.
                batch_size = self.batch_size # Evaluate with this batch size.
            )
         
         #Returns input_ids, masks, token_type_ids    
         return dataloader 


