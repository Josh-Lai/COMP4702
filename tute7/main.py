#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Create the layers of the model here
    
    def forward(self,x):
        return x


def visualise_samples(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15,3))
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

def question3():

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the pixel values
    ])
    mnist_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_subset, val_subset = random_split(mnist_train, [0.7, 0.3], generator=torch.Generator().manual_seed(1))
    # Split the train dataset for the validation daaset
    # Tensor equivalent of the data

    #Data loaders 
    train_loaders = DataLoader(train_subset, shuffle=True)
    test_loaders = DataLoader(mnist_test, shuffle=True)
    val_loader = DataLoader(val_subset, shuffle=True)






if __name__ == "__main__":
    print(torch.__version__)
    print("Cuda Available : {}".format(torch.cuda.is_available()))
    print("GPU - {0}".format(torch.cuda.get_device_name()))
    print("Running week 7")

    question3()
 


