from .seperated_model import SegmentedEarlyExitNetwork
    
       
def seperate_networks(eenet,thresholds,confidence_function=None):
    segmented_networks =[]
    for i in range((len(eenet.network)-1)):
        network = SegmentedEarlyExitNetwork(eenet.network[i],eenet.exits[i],thresholds[i],confidence_function)
        segmented_networks.append(network)
    network  = SegmentedEarlyExitNetwork(eenet.network[-1]).to('cpu')
    segmented_networks.append(network)
    return segmented_networks