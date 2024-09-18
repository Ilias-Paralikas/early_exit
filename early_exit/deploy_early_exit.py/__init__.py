import torch.nn as nn
import torch
import copy



class EarlyExitNetworkSegmentor(nn.Module):
        def __init__(self,network,exit=None,threshold=None,scaler=None):
            super(EarlyExitNetworkSegmentor, self).__init__()
            self.network = copy.deepcopy(network)
            self.exit = copy.deepcopy(exit)
            self.threshold = threshold
            self.scaler = scaler
                
        def forward(self,x):
            x = x.unsqueeze(0)
            x = self.network(x)
            if self.exit != None:
                early_exit = self.exit(x)
                early_exit=  early_exit.squeeze(0)
                confidence = self.scaler(early_exit)
                
                if confidence.max() > self.threshold:
                    return early_exit,True
                return x.squeeze(0),False
            x = x.squeeze(0)
            return x,True


def main():
    return  

if __name__ == "main":
    main()