#! /usr/bin/env python

import torch
import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def create_sequential_model(dim_in:int, dim_out:int, hidden_layer_sizes:List[int]):
  print(dim_in)
  hiddens = [dim_in, *hidden_layer_sizes]
  torch_layers = []
  #Create a linear layer and feed it through a ReLU
  for i in range(len(hiddens)-1):
    torch_layers.append(torch.nn.Linear(hiddens[i], hiddens[i+1]))
    torch_layers.append(torch.nn.ReLU())
  torch_layers.append(torch.nn.Linear(hiddens[-1], dim_out)) #create the output layer
  return torch.nn.Sequential(*torch_layers)

if __name__ == "__main__":
    CIFAR10_train = torchvision.datasets.CIFAR10('CIFAR10_data',download=True,train=True, transform=True)
    CIFAR10_validation = torchvision.datasets.CIFAR10('CIFAR10_data',download=True,train=False, transform=True)
    print(CIFAR10_train.data.shape)
    print(len(CIFAR10_train.classes))
    training_data = (CIFAR10_train.data.reshape((-1,32*32*3))/255.0).astype(np.float32) # flatten the dataset and normalise
    training_labels = np.asarray(CIFAR10_train.targets)
    validation_data = (CIFAR10_validation.data.reshape((-1,32*32*3))/255.0).astype(np.float32) # flatten the dataset and normalise
    validation_labels = np.asarray(CIFAR10_validation.targets)
    
    model = create_sequential_model(32 * 32 * 3, 10, [100,100])

    # Crieterion used to train the model is cross_entropy_loss
    #   Recall that the oss function is the method used to create the weights of the nueral network
    # USe Stochastic gradient descent to optimise the system

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-3
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model

    batch_size = 256
    # THe number of data samples that are passed into the model in a signle foward and backward
    # pass during training. Represents the number of samples the model sees before updating its
    # parameters (in one iteration)

    optimisation_steps = int(1e4)
    # The number of batch sizez that are we train on
    metrics = []
    for i in range(optimisation_steps):
      idx = np.random.randint(0, training_data.shape[0], size = batch_size) # random sample of batch_size indices from 0 to the number of datapoints the dataset has 
      x = training_data[idx,:] # get the datapoints at the sampled indices
      # flattened_x = torch.from_numpy(x.reshape(batch_size,-1)).as # flatten the datapoints
      y_pred = model(torch.from_numpy(x)) # predict the classes of the datapoints)
      loss = criterion(y_pred,torch.from_numpy(training_labels[idx])) # compute the loss by comparing the predicted labels vs the actual labels
      # zero the gradients held by the optimiser
      optimiser.zero_grad()
      # perform a backward pass to compute the gradients
      loss.backward()
      # update the weights
      optimiser.step()
      if i%100==99:
        if i%1000==999:
          train_pred =  model(torch.from_numpy(training_data))
          val_pred =  model(torch.from_numpy(validation_data))
          train_accuracy = torch.mean((train_pred.argmax(dim=1) == torch.from_numpy(training_labels)).float())
          val_accuracy = torch.mean((val_pred.argmax(dim=1) == torch.from_numpy(validation_labels)).float())
          # print the loss every 100 steps
          metrics.append([i,loss.item(),train_accuracy.numpy(), val_accuracy.numpy()])
        print(f'\rEpoch: {i} Loss:{round(loss.item(),2)}', end='')

    metrics = np.asarray(metrics)
    sns.lineplot(x=metrics[:,0],y=metrics[:,1])
    plt.xlabel('step')
    plt.ylabel('training loss')
    plt.show()
    sns.lineplot(x=metrics[:,0],y=metrics[:,2],label='training')
    sns.lineplot(x=metrics[:,0],y=metrics[:,3], label='validation')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.show()
    


