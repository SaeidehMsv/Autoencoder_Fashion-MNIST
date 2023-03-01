# Autoencoder_Fashion-MNIST
implementation of autoencoder and trianing on Fashion-MNIST dataset

Project Description: Autoencoder with Fashion MNIST Dataset and Model Checkpoint for Early Stop and Learning Rate

Autoencoder is an unsupervised deep learning technique used to learn a compressed representation of input data. The Fashion MNIST dataset contains 60,000 grayscale images of 10 different types of clothing items. In this project, we will use autoencoder to learn a compressed representation of the Fashion MNIST dataset.

We will use a deep neural network architecture for the autoencoder model. The input layer will have 784 neurons, which is the size of each image in the Fashion MNIST dataset. We will have two hidden layers with 128 and 64 neurons, respectively. The output layer will have 784 neurons, which is the same as the input layer.

We will train the autoencoder model using the mean squared error loss function and the Adam optimizer. To prevent overfitting, we will use model checkpoint for early stopping and learning rate scheduling. The model checkpoint will save the weights of the best performing model on the validation set during training. Learning rate scheduling will decrease the learning rate if the validation loss does not improve for a certain number of epochs.

After training the autoencoder model, we will use it to encode and decode the images in the Fashion MNIST dataset. We will visualize the reconstructed images and compare them with the original images to evaluate the performance of the autoencoder model.

Overall, this project aims to demonstrate the effectiveness of autoencoder for learning a compressed representation of Fashion MNIST dataset and the importance of using model checkpoint and learning rate scheduling for preventing overfitting.



