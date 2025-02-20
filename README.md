This work was partially support by the ``Trustworthy And Resilient Decentralised Intelligence For Edge Systems (TaRDIS)" Project, funded by EU HORIZON EUROPE program, under grant agreement No 101093006



The function to be exposed is import as follows


`from early_exit.get_early_exit import create_networks`

as used as such
How to Use:

1. Define your model (optionally train it on your dataset), as you would in a normal torch model.

2. Define a class that splits your model into nn.Sequentail Subbolocks that, in turn all belong to an ```nn.Sequential``` Container, named "self.net" (see demo.ipynb, SplitModel class) 

3. Define the exit layers. (see demo.ipynb, ExitLayers class). The extis can be defined as a nn.Sequential, just like the split model.

4. For each exit, infer the output shape and define the exits manually. (see next cell at demo).

5. Define the exits and add them to a ```nn.ModuleList```, witht the sequentials (note that in the demo, the exit.network (the nn.sequential part of the model is passed))

6. Call the ```EarlyExitNetwork``` class, with parameters the seperated_models.net and the exits.
**Important**  : You should provide the sequential part of the model and the exits, not the whole model. 
