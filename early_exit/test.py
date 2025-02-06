import torch
def test_all_exits_accuracy(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)
    model.eval()
    with torch.no_grad():     
        images,_ = next(iter(dataloader))
        images  = images.to(device)
        outputs=  model(images)
        batch_size = outputs.shape[0]
        num_exits = outputs.shape[1]
        correct = [0] * num_exits
        total = 0
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
             
                        
            for batch_number, batch_output in enumerate(outputs):
                for exit in range(num_exits):
                    answer = torch.argmax(batch_output[exit],dim=0)
                    if answer == labels[batch_number]:
                        correct[exit] += 1
                total += 1
                    
        return correct,total     
        # accuracies = {}
        # for i, (c, t) in enumerate(zip(correct, total)):
        #     print(f'Accuracy of exit {i}: {100 * c / t}%')
        #     accuracies[f'exit_{i}'] = 100 * c / t
        # return accuracies
