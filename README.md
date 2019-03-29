# element encoder
Autoencoder neural network to compress properties of atomic species into a vector representation

## Prerequisites
Required packages
```
numpy
tensorflow
```

## Getting Started
To download this project run the following command
```
git clone https://github.com/jeherr/element-encoder.git
```
You can run with the default parameters with the following command
```
python train.py
```
As the network trains, it will run a test after every 5th epoch by printing the original features fed into the network and the corresponding reconstructed features for a random sampling of 10 different atomic species. After training has finished, the best checkpoint is reloaded, a test step is rerun for that checkpoint, and the resulting latent space vectors are written to an emodes.dat file where the rows correspond to elements as listed in the element_data.py file and the columns are the latent vector dimensions.

## Training your own model
To train your own model with different hyperparameters, you can adjust variables passed to the network in train.py. Most parameters passed in should be ints, but learning_rate should be a float (generally in the range of 1.e-1 to 1.e-6) and hidden_layers should be a list of ints where both the encoder and decoder will have number of hidden layers equal to the length of the list and the values in the list will correspond to the number of neurons in each layer (e.g. hidden_layers=[128, 128] will result in an encoder and decoder which both have two hidden layers with 128 neurons in each layer).

Since our data set is the elements up to Bi excluding f-block, then a typical epoch would only have 69 data points which may only lead to 1 or 2 batches per epoch. Instead, batches_per_epoch allows to decide how many batches you want to train on before 1 "epoch" completes. Batches are made by randomly selecting batch_size number of atomic species with repeats allowed in the same batch.  

The data set used is in element_data.py should you wish to add, remove, or alter any data used for your own training set.

## Citing this work
A preprint of this paper is available at https://arxiv.org/abs/1811.00123 for citation. Link to peer-reviewed publication forthcoming.
