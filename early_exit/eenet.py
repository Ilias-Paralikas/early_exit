import torch
import torch.nn as nn
from copy import deepcopy





class EarlyExitNetwork(nn.Module):
    '''
    
    '''
    def __init__(self,separated_network,exit_layers,device=None):
        super(EarlyExitNetwork, self).__init__()
        
        assert len(exit_layers) == len(separated_network)-1
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = deepcopy(separated_network).to(device)
        

        self.exits = exit_layers
        self.len = len(self.network)

    def forward(self,x,exit_chosen=None):
        if exit_chosen is None:
            outputs = []
            for i in range(self.len-1):
                x = self.network[i](x)
                early_exit = self.exits[i](x)
                outputs.append(early_exit)
                
            x = self.network[-1](x)
            outputs.append(x)
            return torch.stack(outputs, dim=1)
        else:
            assert 0<=exit_chosen<self.len
            for i in range(exit_chosen+1):
                x = self.network[i](x)

            if exit_chosen ==self.len-1:
                return x
    
            return self.exits[exit_chosen](x)

    