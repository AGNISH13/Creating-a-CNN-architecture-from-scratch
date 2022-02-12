# Creating-a-CNN-architecture-from-scratch

## Project Description
This is a python-based project in which a basic CNN (Convolution Neural Network) architecture has been created from scratch to classify the digits of the MNIST dataset, giving a different Training and Validation accuracy for each epoch, the average of which is around 98%.

## Dataset Description
MNIST ("Modified National Institute of Standards and Technology") is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. The dataset is available at: http://yann.lecun.com/exdb/mnist/    
Here, the MNIST dataset has been imported to our python file from `torchvision.datasets`

## Classes of Division
The digits are finally classified into 10 classes, i.e., from 0 to 9.  

## Architecture of the Convolution Neural Network
The CNN created here is having 6 layers, namely:    
- `Convolutional Layer 1`  
- `Inception Block`  
- `Convolutional Layer 2`
- `Convolutional Layer 3`    
- `Convolutional Layer 2`
- `Max-Pool Layers`
- `Fully Connected Layers`

Pictorial representation for our CNN model:



## Train-Validation Learning Curve
Train-Validation Curve is a popular method to helps us confirm normal behavioural characteristics of model over increasing number of epochs. 
 
The CNN model used here has been trained over `20` epochs with batch_size of `60` for Training set and `50` for validation set.
-     Train-Validation Curve for our CNN model
     ![image](https://user-images.githubusercontent.com/84792746/153714235-9f53f2f6-4f74-49a2-87ac-f4960e6af32a.png)

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`
        
## Run the following for training and validation :
  
   `python main.py`
