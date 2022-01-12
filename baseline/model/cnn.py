import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class CNNClassifier(nn.ModuleList):
    def __init__(self,vocab_size=None, embed_dim=300, ):
      super(CNNClassifier, self).__init__()
      
      self.embed_dim = embed_dim
      self.num_filters = num_filters=[100, 100, 100,100]
      #Filter size acts as sliding window of tokens, e.g., bi-grams, tri-grams etc
      self.filter_sizes=[2, 3, 4, 5]
      self.num_classes = 1
      
      #contains a “+1” this because we are considering the index that refers to the padding, in this case it is the index 0.
      self.embedding = nn.Embedding(num_embeddings=vocab_size+1,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0)    
      
      self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])
      
      self.fc = nn.Linear(np.sum(self.num_filters), self.num_classes)
      self.drop_out = nn.Dropout(p=0.5)
      
      
    def forward(self, input_ids):   
        
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()
        
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        
        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
           for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.drop_out(x_fc))
        
        logits = torch.sigmoid(logits) #sigmoid as we use BCELoss
        
        return logits

