from ..deploy_early_exit import EarlyExitNetworkSegmentor
from ..train_early_exit import EarlyExitNet,train_early_exit_network,test_all_exits_accuracy,seperate_networks


def create_networks(model,
                    input_shape,
                    thresholds,
                    neurons_in_exit_layers,
                    epochs,
                    train_dataloader,
                    test_dataloader,
                    optimizer,
                    optimizer_parameters,
                    criterion,
                    training_method,
                    scaler = None
):
    eenet = EarlyExitNet(model.net,input_shape,thresholds,neurons_in_exit_layers,scaler)
    train_losses = train_early_exit_network(eenet,epochs,train_dataloader,criterion,optimizer,optimizer_parameters,training_method)
    test_accuracies  = test_all_exits_accuracy(eenet,test_dataloader)
    networks= seperate_networks(eenet)
    return networks,train_losses,test_accuracies
