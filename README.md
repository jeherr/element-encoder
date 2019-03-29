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
As the network trains, it will run a test after every 5th epoch by printing the original features fed into the network and the corresponding reconstructed features for a random sampling of 10 different atomic species. After training has finished, the best checkpoint is reloaded, a test step is rerun for that checkpoint, and the resulting latent space vectors are written to an emodes.dat file.

## Training your own model
To train your own model run
```
python train.py train_set.pkl
```
which uses the same hyperparameters laid out in the paper and stops after 1000 epochs. The model is evaluated on the validation set after every 5 epochs and prints out a random sample of 10 true formation energies and the corresponding predictions, along with mean absolute errors, mean signed errors, and root mean square errors over the validation set. Only saves a new checkpoint if the evaluation loss is lower than the last saved checkpoint. After training finishes, the best checkpoint is reloaded and the errors are evaluated over the test set. Trained model create a new directory with the data and time the model was started. To use your own model with evaluate.py, replace the default model directory with the directory for your newly trained model in the following line.
```
model = network_model.NNModel(name="ElpasEM_Thu_Mar_21_13.16.26_2019")
```

## Citing this work
A preprint of this paper is available at https://arxiv.org/abs/1811.00123 for citation.


## Acknowledgments
Thanks to the work linked below for the data set of elpasolites and formation energies used here to train the model.
https://doi.org/10.1103/PhysRevLett.117.135502
