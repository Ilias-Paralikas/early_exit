How to Use:

1. Define you model (optinal train it) on your dataset. 

2. Define a class that splits your model into nn.Sequentail Subbolocks that, in turn all belong to an nn.Sequential Container, named "self.net" (see demo.ipynb, SplitModel class)

3. Define the parameters the following parameters 


| Parameter               | Type                                      | Description                                                                 |
|-------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| `model`                 | `Split Model `                            | The split model as defined in step 2                                        |
| `input_shape`           | `Tuple   `                                  | Shape of the input data (e.g., `(1, 3, 32, 32)`).                           |
| `thresholds`            | `List of floats    `                        | Threshold values for early exits, it exceded the exit is taken              |
| `neurons_in_exit_layers`| `List of lists of integers   `              | Number of neurons in each exit layer.                                       |
| `epochs`                |` Integer     `                              | Number of epochs to train the model.                                        |
| `train_dataloader`      | `torch.utils.data.DataLoader`             | DataLoader for training data.                                               |
| `test_dataloader`       | `torch.utils.data.DataLoader`             | DataLoader for test data.                                                   |
| `optimizer`             | `torch.optim`                       | Optimization algorithm for training(e.g., `torch.optim.SGD`)                |
| `optimizer_parameters`  | `dict`                                | Parameters to configure the optimizer (e.g., `{'lr': 0.001}`).              |
| `criterion`             | `torch.nn `                            | Loss function to train the model .(e.g.,`torch.nn.CrossEntropyLoss()`)      |
| `training_method`       | `String` or `Integer`                         | Method of training (`'whole_network'`, `'all_exits'`, or one exit layer)    |

NOTE on training method.


The whole network option will train the whole network as well as the exits
the all_exits option will train ONLY the added exits (works well for pretrained networks that we don't want to hurt the performance of the body)
If an integer is passed as a parameter, it will train the exit specified. Obviously, the number must not exceed the number of exits.
