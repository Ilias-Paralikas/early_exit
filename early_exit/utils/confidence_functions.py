
import torch
'''
When arriving at an exit, we have to quantify how confident the network is in it's answer.
If the output of the network are unbound scalers, it is hard to know beforehand their range and compare them
with a confidence threshold, defined before training (or even after).
For this reason, we use softmax temperature, in roder to scale the output of the network to a range [0,1].
This way, we can compare the output of the network with a threshold, and decide if we should exit or not.

The codes below allow the user to define their own scaler, or use the softmax_temperature function, if none is passed.
Basically, saves us 5 lines of code everytime we need to create a new instance of confidence function,
as well as making sure that we are using the same function in all the places we need it.
'''

def softmax_temperature(logits, temperature=1):
    logits = logits / temperature
    return torch.softmax(logits, dim=0)


def get_confidence_function(function_name=None):
    available_functions ={
        'softmax_temperature':softmax_temperature
    }
    return available_functions[function_name]