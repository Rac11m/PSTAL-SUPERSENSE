import torch 
import torch.nn as nn

'''
* Le classifieur MLP: composé de une ou plusieurs couches linéaires (nn.Linear) suivies de fonctions d’activation
non-linéaires (p.ex. nn.ReLu). 

* La première couche dense prend en entrée l’embedding contextuel du mot à classifier.
    ** Sa dimension dépend du modèle pré-entraîné utilisé (camembert, distilbert, . . .) et peut être obtenue avec config.hidden_size. 
* La dernière couche aura comme fonction d’activation le softmax (implicite, comme d’habitude).


---- Il n’est pas nécessaire de faire du padding comme on ferait dans RNN.
'''

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