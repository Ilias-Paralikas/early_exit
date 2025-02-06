import torch
def train_whole_network(model,loader,criterion,optimizer,device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img,mask in loader:
        img = img.to(device)
        mask = mask.to(device)
        output = model(img)
        loss = criterion(output,mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        