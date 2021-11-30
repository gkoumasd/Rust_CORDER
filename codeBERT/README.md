## CodeBERT



## Data Preparation

1. Download the data from repo https://github.com/yijunyu/cargo-geiger/blob/datasets/datasets/unique-indented.tar.bz2?raw=true

2. Run the script split_data to split them in train/val/test. ```python split_data.py``` 
 

After these steps, you can see the data in the treesitter_rust_train_test_val folder, splitted into 3 subfolders train\test\val. 


3. Preprocess the data

    
    - ```source process_data.sh```

This step will process the data and save input_ids, attention_masks, and labels to pickle files for the train, val, and test subdatasets.

