import torch
def train_whole_network(model,loader,criterion,optimizer,device=None):
    def create_onehot_repeated_tensor(labels_batch, n_classes, repeats):
        # Get batch size from labels
        batch_size = labels_batch.shape[0]
        
        # Create empty tensor to store results
        result = torch.zeros(batch_size, repeats, n_classes, dtype=torch.float)
        
        # For each item in the batch
        for i in range(batch_size):
            # Create one-hot encoding
            one_hot = torch.zeros(n_classes, dtype=torch.float)
            one_hot[labels_batch[i]] = 1  # Remove the [0] indexing since labels are 1D
            
            # Repeat for specified number of times
            result[i] = one_hot.unsqueeze(0).repeat(repeats, 1)
        
        return result
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for img,labels in loader:
        img = img.to(device)
        labels = labels.to(device)
        output = model(img)
        
        number_of_exits =output.shape[1]
        number_of_classes = output.shape[2]
        labels = create_onehot_repeated_tensor(labels,number_of_classes,number_of_exits)
      
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()