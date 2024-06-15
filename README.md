Classification Object in CIFAR10 Dataset using CNN Model
This repository contains the implementation of a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is divided into 50,000 training images and 10,000 test images.

Dataset
The CIFAR-10 dataset can be downloaded from the official website here. (https://www.cs.toronto.edu/~kriz/cifar.html)

Model Architecture
The CNN model implemented in this project consists of several convolutional layers, followed by max-pooling layers, and fully connected layers. The architecture is designed to effectively capture spatial hierarchies in the image data, improving classification performance.

Features

Data Preprocessing: Includes normalization and data augmentation to improve model generalization.

Model Training: Utilizes Keras with TensorFlow backend to build and train the CNN model.

Performance Evaluation: Provides evaluation metrics including accuracy and loss plots, as well as confusion matrix for detailed performance analysis.

Saved Model: The trained model can be saved and loaded for future predictions.

Installation
To run this project, you need to have Python installed along with the following libraries:

TensorFlow

Keras

NumPy

Matplotlib

Scikit-learn
