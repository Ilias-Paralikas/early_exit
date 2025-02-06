import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy





class EarlyExitNetwork(nn.Module):
    '''
    
    '''
    def __init__(self,separated_network,exit_layers,thresholds,device=None):
        super(EarlyExitNetwork, self).__init__()
        
        assert len(exit_layers) == len(separated_network)-1
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.thresholds  = thresholds
        self.network = deepcopy(separated_network).to(device)
        

        self.exits = exit_layers
        self.len = len(self.network)

    def forward(self,x):
        outputs = []
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            outputs.append(early_exit)
            
        x = self.network[-1](x)
        outputs.append(x)
        return torch.stack(outputs, dim=1)

    def segmented_forward(self,x):
        x=  x.unsqueeze(0)
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit= early_exit.squeeze(0)
            
            if early_exit.max() > self.thresholds[i]:
                return early_exit,i

        x = self.network[-1](x)
        x = x.squeeze(0)
        return x,i+1
    
    def specific_exit_forward(self,x,exit):
        assert exit<=self.len-1
        x = self.network[0](x)
        for j in range(exit):
            x = self.network[j+1](x)
        if exit <self.len-1:
            x = self.exits[exit](x)
        return x