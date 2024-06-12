# GraphRNN Supported Upcycling of Wood Waste Into Building Components

This code is part of our bachelor's thesis in Computer Engineering at Aarhus University.

## important files 
### generator.py
It contains the code with all the logic for generating perfect packing solutions.
### data.py
It contains the code responsible for building and loading the datasets.
### models.py 
It contains the code for our machine-learning model. 
- Training
- Testing
- Inference

## Generating a packing solution using the models.py file

1. Change the "mode" variable to "test".
2. Change the "model_dir_name" to "model_x", where "x" is the model you want to use.
3. Change the "data_graph_size" to the corresponding graph size. There has to be a model of that size available.
4. Run the Python file.

- The dataset each model has been trained with is inside the model folder if testing on the training data.
- Remember to change the "test" variable to false, as this changes to using the training part of the dataset.
