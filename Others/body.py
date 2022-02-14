

from Others import CNN_architecture

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.utils.data as data_utils
import torchvision.transforms as transf
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import time

def main_body():


    # Configuring Device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameter tuning

    epoch_no=200
    l_rate = 0.00001
    batch_size_tr = 60
    batch_size_val = 50

    # Performing Data Augmentation for increased accuracy

    train_transform = transf.Compose([
        transf.Resize((224,224)),
        transf.ColorJitter(brightness=0.5),
        transf.ToTensor()
    ])

    val_transform = transf.Compose([
        transf.Resize((224,224)),
        transf.ToTensor()
    ])

    # Loading the dataset in the system

    ds = MNIST(root='data/', download=True,train=True,transform = train_transform)
    train_ds, val_ds = random_split(ds,[50000,10000])

    # Un-comment the following part in order to reduce the dataset
    #tr_indices = torch.arange(25000)
    #tr_new = data_utils.Subset(train_ds, tr_indices)
    #val_indices = torch.arange(5000)
    #val_new = data_utils.Subset(val_ds, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size_tr, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=True, num_workers=2, drop_last=True)

    # Training our CNN model 

    def train_model(model, criterion, optim, epoch_no):
        
        since = time.time()
        train_loss_values = []
        val_loss_values = []
        best_acc= 0.0
        for epoch in range(epoch_no):
            running_loss = 0.0
            running_acc = 0.0
            model.train()
            for images,labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    outputs = model(images)
                    _ ,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optim.step()
                optim.zero_grad()

            # Calculating and printing all Statistics

            running_loss += loss.item()*batch_size_tr
            running_acc += torch.sum(preds==labels)
            running_val_loss, running_val_acc = model_val(model, criterion, optim)
            epoch_train_loss = running_loss/len(train_ds)
            epoch_train_acc = running_acc.double()/len(train_ds)
            print("Epoch: {}".format(epoch+1))
            print('-'*10)
            print('Train Loss: {:.4f}   Train Acc: {:.4f}'.format(epoch_train_loss,epoch_train_acc))
            epoch_val_loss = running_val_loss/len(val_ds)
            epoch_val_acc = running_val_acc.double()/len(val_ds)
            print('Val Loss: {:.4f}   Val Acc: {:.4f}'.format(epoch_val_loss,epoch_val_acc))
            print()
            train_loss_values.append(epoch_train_loss)
            val_loss_values.append(epoch_val_loss)
            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc

        # Printing Time Elapsed, Best Validation Accuracy and Loss vs Epoch Plot

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best model has validation accuracy: {}".format(best_acc))
        plt.plot(range(epoch_no),np.array(train_loss_values),'b',label='Train Curve')
        plt.plot(range(epoch_no),np.array(val_loss_values),'g',label='Validation Curve')
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.legend()

    # Validating our CNN model

    def model_val(model, criterion, optim):
        model.eval()
        running_val_loss = 0.0
        running_val_acc = 0.0
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _ ,preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)
            running_val_loss += loss.item()*batch_size_val
            running_val_acc += torch.sum(preds==labels)
        return running_val_loss, running_val_acc

    # Specifying model and performing Loss calculation and Optimization

    model = CNN_architecture.my_model().to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optim = torch.optim.Adam(model.parameters(), lr = l_rate)

    # Training and Evaluation

    train_model(model, criterion, optim, epoch_no)