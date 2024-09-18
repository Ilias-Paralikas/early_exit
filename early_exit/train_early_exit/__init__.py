import torch
import torch.nn as nn
import numpy as np
from ..deploy_early_exit import EarlyExitNetworkSegmentor




def softmax_temperature(logits, temperature=1):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)


class EarlyExitNet(nn.Module):
    def __init__(self,network,input_shape,thresholds,neurons_in_exit_layers = [[1024,1024],[1024,1024]],scaler=None):
        def get_output_flattened( network, input_shape,device):
            x = torch.rand(input_shape).to(device)  # Add the batch size dimension here
            return network(x).view(x.size(0), -1).size(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EarlyExitNet, self).__init__()  
        self.thresholds  = thresholds
        self.network = network.to(device)
        x = torch.rand(input_shape).to(device)
        
        if isinstance(neurons_in_exit_layers,int):
            layers  =  neurons_in_exit_layers
            neurons_in_exit_layers = [[layers] for _ in range(len(network)-1)]
        self.output_shape  =  network(x).size(1)
        
        self.len = len(network)
        assert len(neurons_in_exit_layers) == len(network)-1

        self.exits = nn.ModuleList([])
        for i, net in enumerate(network[:-1]):
            layers = [nn.Flatten()]
            last_layer_size  = get_output_flattened(net,input_shape,device)
            for next_layer_size in neurons_in_exit_layers[i]:
                  
                layers.append(nn.Linear(last_layer_size, next_layer_size))
                layers.append(nn.ReLU())
                last_layer_size = next_layer_size
            layers.append(nn.Linear(last_layer_size, self.output_shape))
            
            input_shape  = net(torch.rand(input_shape).to(device)).size()
        
            self.exits.append(nn.Sequential(*layers).to(device))
            
        
        if scaler is None:
            self.scaler  = softmax_temperature
        else:   
            self.scaler = scaler
                    
    def forward(self,x):
        outputs =[]
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit = self.scaler(early_exit)
            outputs.append(early_exit)
            
        x = self.network[-1](x)
        outputs.append(x)
        return outputs
    def segmented_forward(self,x):
        x=  x.unsqueeze(0)
        for i in range(self.len-1):
            x = self.network[i](x)
            early_exit = self.exits[i](x)
            early_exit= early_exit.squeeze(0)
            early_exit = self.scaler(early_exit)
            
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

    
  
def train_early_exit_network(  model,
            epochs,
            trainloader,
            criterion,
            optimizer,
        optimizer_params, 
        training_method,
             exit_weights= None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters you want to train
    if training_method == 'whole_network':
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optimizer(model.parameters(), **optimizer_params)
    elif training_method == 'all_exits':
        for exit in model.exits:
            for param in exit.parameters():
                param.requires_grad = True
        optimizer = optimizer([param for exit in model.exits for param in exit.parameters()], **optimizer_params)
    elif isinstance(training_method, int):
        if training_method ==model.len-1:
            for param in model.network[-1].parameters():
                param.requires_grad = True
            optimizer = optimizer(model.network[-1].parameters(), **optimizer_params)
        else:
            for param in model.exits[training_method].parameters():
                param.requires_grad = True
            optimizer = optimizer(model.exits[training_method].parameters(), **optimizer_params)
    else:
        raise ValueError('Invalid training method, please choose from "whole_network", "all_exits" or an integer, to specify which exit to train')

    if exit_weights is None:
        exit_weights = [1/model.len for _ in range(model.len)]
    else :
        assert len(exit_weights) ==model.len
        exit_weights =  [i/sum(exit_weights) for i in exit_weights]
    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if isinstance(training_method, int):
                output = model.specific_exit_forward(inputs,training_method)
                loss =criterion(output, labels)
            else:
                outputs = model(inputs)
                loss = 0
                for i in range(len(outputs)):
                    loss += exit_weights[i] *criterion(outputs[i], labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        running_loss= running_loss/len(trainloader)
        losses.append(running_loss)
        
        print(epoch,'loss: ',running_loss)
        running_loss = 0.0
    return losses
       



def test_all_exits_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    model.eval()
    with torch.no_grad():     
        images,_ = next(iter(dataloader))
        images  = images.to(device)
        num_exits = len(model(images))
        correct = [0] * num_exits
        total = [0] * num_exits
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            for i, output in enumerate(outputs):
                _, predicted = torch.max(output.data, 1)
                total[i] += labels.size(0)
                correct[i] += (predicted == labels).sum().item()
        
        
        accuracies = {}
        for i, (c, t) in enumerate(zip(correct, total)):
            print(f'Accuracy of exit {i}: {100 * c / t}%')
            accuracies[f'exit_{i}'] = 100 * c / t
        return accuracies


            
            
def segmented_test_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    exits_chosen =  np.zeros(model.len)
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            for img,lbl in zip(images,labels):
                output,exit = model.segmented_forward(img)  # Use the segmented_forward function
                prediction = torch.argmax(output).item()  # Get the predictions from the output
                total += 1
                correct += (prediction == lbl)
                exits_chosen[exit] +=1
    return correct / total ,exits_chosen

       
def seperate_networks(eenet):
    segmented_networks =[]
    scaler  = eenet.scaler
    for i in range((len(eenet.network)-1)):
        network = EarlyExitNetworkSegmentor(eenet.network[i],eenet.exits[i],eenet.thresholds[i],scaler)
        segmented_networks.append(network)
    network  = EarlyExitNetworkSegmentor(eenet.network[-1]).to('cpu')
    segmented_networks.append(network)
    return segmented_networks