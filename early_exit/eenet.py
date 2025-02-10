import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EarlyExitNetwork(nn.Module):
    '''
    This is a class that taken A SEPARATED network and a modulelist of exit layers and creates an early exit network
    The separated network  is the original network that the user has manually separated and placed into a
    nn.Sequential container.
    Each exit will be placed at the end of each part of the sequential network.
    Dimension matching is left to the user.
    The last exit is the output if part of the original network so the exits provided have to be n-1 where n is the number of parts in the separated network.
    '''
    
    def __init__(self,
                 separated_network: nn.Sequential,
                 exit_layers: nn.ModuleList,
                 device: Optional[torch.device] = None):
        super(EarlyExitNetwork, self).__init__()
        # if the network is broken up in n different pieces, the exit_layers should be  n-1
        # as the last exit is the output layers of the original model
        assert len(exit_layers) == len(separated_network)-1
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.network = deepcopy(separated_network).to(device)
        self.exits = exit_layers
        self.len = len(self.network)

    def forward(self,x,exit_chosen=None):
        '''
        When not providing an exit, all of them will be used simultaneously.
        This is mostly used if one want to train or test the whole network.
        '''
        if exit_chosen is None:
            outputs = []
            for i in range(self.len-1):
                # itterate over the core network
                x = self.network[i](x)
                # store the exit in the outputs
                early_exit = self.exits[i](x)
                outputs.append(early_exit)
            
            # the last exit is part of the original network
            x = self.network[-1](x)
            outputs.append(x)
            # very carefull with shapes and dimensions, dim =0 is the batch so we concat on dim=1
            return torch.stack(outputs, dim=1)
        else:
            # if we want only one exit, make sure we ae between 0 and the number of exits
            assert 0<=exit_chosen<self.len
            # itterate over the network until we reach the exit we want
            for i in range(exit_chosen+1):
                x = self.network[i](x)
            # if that exit is the last layer, we just return as it is part of the original network
            if exit_chosen ==self.len-1:
                return x
            # else we return the exit
            return self.exits[exit_chosen](x)

    