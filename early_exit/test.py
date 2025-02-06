import torch
def test_all_exits_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    model.eval()
    with torch.no_grad():     
        images,_ = next(iter(dataloader))
        images  = images.to(device)
        outputs=  model(images)
        num_exits = outputs.shape[1]
        correct = [0] * num_exits
        total = 0
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            batch_size =outputs.shape[0]

            for b in range(batch_size):
                for e in range(num_exits):
                    prediction = torch.argmax(outputs[b][e],dim=0)
                    if prediction.item() == labels[b]:
                        correct[e] +=1

            total += batch_size
        return correct,total
   