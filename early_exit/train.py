from tqdm import tqdm
import torch
def train_model(model,loader,criterion,optimizer,epochs=1,exit_chosen=None,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    loop = tqdm(loader)
    images,_ = next(iter(loader))
    images = images.to(device)
    output = model(images)
    if exit_chosen is None:
        num_exits = output.shape[1]
    num_classes = output.shape[-1]
    for epoch in range(epochs):
        for images,labels in loop:
            images= images.to(device)
            
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            output = model(images,exit_chosen=exit_chosen)

            if exit_chosen is None:
    
                labels = labels.unsqueeze(1).repeat(1, num_exits, 1)  # Repeat along new dimension
                labels = labels.view(-1,num_exits*num_classes)
                output =output.view(-1,num_exits*num_classes)
            labels = labels.to(device)
            loss  =criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(
                loss=loss.item()
            )
