import os#
from tqdm import *
import random
from sklearn.model_selection import train_test_split
import shutil


def split_data(category):

    directory = 'data/' + category
    files_lst = []
    
    for dirpath , subdirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.endswith(".rs"):
                files_lst.append(os.path.join(dirpath, file))
    
    
    random.shuffle(files_lst)
    
    
    # In the first step we will split the data in training and remaining dataset
    
    X_train, X_rem = train_test_split(files_lst, train_size=0.8, random_state=42)
    
    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test = train_test_split(X_rem, test_size=0.5, random_state=42)
    
    
    for file in X_train:
        shutil.copy2(file, 'treesitter_rust_train_test_val/train/' + category)
    
    for file in X_test:
        shutil.copy2(file, 'treesitter_rust_train_test_val/test/' + category)  
        
    for file in X_valid:
        shutil.copy2(file, 'treesitter_rust_train_test_val/val/' + category)  
        
        
split_data('safe')
split_data('unsafe')        