# CIFAR-10-Image-Colorization
Explore the vibrant world of CIFAR-10 image colorization with Depp Learning.

**Image colorization** is the process of adding color to grayscale images, a fascinating task that can be effectively addressed using deep learning techniques. This project focuses on developing and training deep learning models to automatically colorize grayscale images, leveraging the renowned CIFAR-10 dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Model Training](#model-training)
- [Results and Visualization](#results-and-visualization)


## Introduction

In this project, we implement and train convolutional neural network (CNN) models for image colorization.The main notebook also contains a U-net model for implementing skip connection technique, additionally to enhance the performance and accuracy of colorizing residual blocks were applied to the U-net model. Here's an overview of the key components and features of the project:

- **`model.py`**: This module contains the implementation of various CNN models for image colorization. You can choose from different model architectures, including a basic CNN, a U-Net, and a ResNet. These models are designed to predict color categories for grayscale input images.

- **`torch_helper.py`**: This helper module provides a set of functions for data processing, data loading, and training. It simplifies tasks like generating data batches, converting data to PyTorch tensors, calculating loss, and running the validation step during training.

- **`train.py`**: The primary training script allows you to train image colorization models. It utilizes the models defined in `model.py`, handles the CIFAR-10 dataset, and executes the training loop. You can customize various training parameters to suit your specific needs.

- **`utils.py`**: A collection of utility functions used to load and preprocess CIFAR-10 data, as well as to visualize results. These utilities assist with data loading, data preprocessing, and visualizing image colorization outputs.

## Dataset

For this task, we utilized the CIFAR-10 dataset, specifically the "automobile" class. The CIFAR-10 dataset contains 60,000 color images with dimensions of 32x32 across 10 classes. The dataset is split into 50,000 training images and 10,000 test images. We convert the original RGB images to grayscale images for input and create labels corresponding to the color of each pixel within each input image.

## Getting Started

To get started with this project, you'll need Python and PyTorch installed on your system. We recommend using Anaconda or virtual environments to manage your Python environment.
Ensure you have a GPU available for training, as the tasks may require significant computational resources. If you're using Colab, remember to enable GPU support.

## Model Training
Several models were implemented with bellow training parameters:
- Number of Filters (NF)
- Learning Rate (LR)
- Kernel Size
- Number of Epochs

### Base Model (CNN)
The base model is initially trained to address the image colorization task. The network architecture is as follows:  

<img width="350" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/780af039-76a2-4dbd-aebb-fabb42945fb7">  
Base Model Architecture  


### Custom U-Net Model
We introduce skip connections to improve the model's performance. The architecture is based on U-Net and includes skip connections.  

<img width="453" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/8c2555d3-86bc-4369-b700-7b70eb79e699">  

Custom U-Net Architecture  


### U-Net with Residual Block
An extra point task involves adding Residual Blocks to DownConv, UpConv, and Bottleneck layers.  

<img width="257" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/8cb7ee0c-6b1b-47f1-b8ba-7dbff7f15c9b">  
Residual Block  


## Results and Visualization
![image](https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/d41c1835-71de-4270-b565-917017551fcc)  
colorized figures  

<img width="200" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/f0c7c6a7-bbef-4b94-879d-3c43cdc6f018">  

value of loss during training CNN model  

<img width="218" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/dfa13102-0d77-46da-b9f1-fadb96073fe7">  

value of loss during training U-net model  

<img width="301" alt="image" src="https://github.com/zkhotanlou/CIFAR-10-Image-Colorization/assets/84021970/1af74401-3080-4654-aa40-4ab394c414a2">  

value of loss during training ResNet model

## References
- Olaf Ronneberger, Philipp Fischer, Thomas Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in International Conference on Medical Image Computing and Computer-Assisted Intervention, 2015.
- Stanford University Convolutional Neural Networks Tutorial

