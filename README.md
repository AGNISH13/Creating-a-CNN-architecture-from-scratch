# Creating-a-CNN-architecture-from-scratch

## Project Description
This is a python-based project in which a basic CNN (Convolution Neural Network) architecture has been created from scratch to classify the digits of the MNIST dataset, giving a different Training and Validation accuracy for each epoch, the average of which is around 98%.

## Dataset Description
MNIST ("Modified National Institute of Standards and Technology") is the de facto “Hello World” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.    
Here, the MNIST dataset has been imported to our python file from `torchvision.datasets`

## Classes of Division
The digits are finally classified into 10 classes, i.e., from 0 to 9.  

## Architecture of our Convolution Neural Network


## Train-Validation Learning Curve
Train-Validation Curve is a popular method to helps us confirm normal behavioural characteristics of model over increasing number of epochs. 
 
Our CNN model has been trained over `20` epochs with batch_size of `60` for Training set and `50` for validation set.
-     Train-Validation Curve for our CNN model
     ![image](https://user-images.githubusercontent.com/89198752/153136792-b68cb600-5f30-4ddc-bb78-3dee08e0e2f9.png)

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are  `matplotlib`, `numpy`, `torch` and `torchvision`. All these dependencies can be installed just by the following command line argument
- pip install `requirements.txt`
        
## Run the following for training and validation :
  
   `python main.py`
