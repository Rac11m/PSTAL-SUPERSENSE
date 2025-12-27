import torch 
import torch.nn as nn


class SUPERSENSE_model(nn.Module):
    def __init__(self, emb_dim, output_size):
        '''
        emb_dim: correspond to config.hidden_size of the bert model
        output_size: correspond to number of supersense tags + 1 (for * tag)
        '''
        super().__init__()
        
        self.linayer = nn.Linear(emb_dim, emb_dim*2)
        self.nonlinayer = nn.ReLU()        
        self.dropout = nn.Dropout(0.1)
        self.decision = nn.Linear(emb_dim*2, output_size)
    
    def forward(self, emb_context):
        x = self.linayer(emb_context)
        x = self.nonlinayer(x)
        x = self.dropout(x)
        return self.decision(x)