import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class BiLSTMClassifier(nn.ModuleList):
    def __init__(self, voc_size, drop=0.3, hidden_dim=128, num_layers=1,  embed_dim=300, ):
      super(BiLSTMClassifier, self).__init__()
      
      
      self.num_layers = num_layers
      self.hidden_dim = hidden_dim
      self.voc_size = voc_size
      self.dropout = nn.Dropout(drop)
    
      
      self.embedding = nn.Embedding(self.voc_size+1, self.hidden_dim, padding_idx=0)
      
      self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
      
      
      self.fc1 = nn.Linear(in_features=self.hidden_dim*2, out_features=128)
      self.fc2 = nn.Linear(128, 1)
      

    
    def forward(self, x):
        
        #h = torch.zeros((self.num_layers*2, x.size(0), self.hidden_dim)).to(self.device)
        #c = torch.zeros((self.num_layers*2, x.size(0), self.hidden_dim)).to(self.device)
        
        out = self.embedding(x)
        
        out, (hidden, cell) = self.lstm(out)
        out = self.dropout(out)
        
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        
        out = torch.sigmoid(self.fc2(out))

        return out