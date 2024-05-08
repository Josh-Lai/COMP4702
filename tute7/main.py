#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super(MLP, self).__init__()
        # Create the layers of the model here
        # Output 10x1
        # Input 28x28

        # Set up the first layer
        hidden_layers = 512
        self.fc1 = nn.Linear(dim_in, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, hidden_layers)
        self.out = nn.Linear(hidden_layers, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Parse this through the layer
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x


def visualise_samples(dataset, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()


def question3():

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        # Normalize the pixel values
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(
        "data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(
        "data", train=False, download=True, transform=transform)

    test_subset, val_subset = random_split(
        mnist_test, [0.7, 0.3], generator=torch.Generator().manual_seed(1))
    train_subset = mnist_train
    # Split the train dataset for the validation daaset
    # Tensor equivalent of the data

    # Data loaders
    # Load 750 random samples
    num_samples = 750
    sampler = SubsetRandomSampler(range(num_samples))
    batch_size = 1
    train_loaders = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler)
    test_loaders = DataLoader(test_subset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
    # Get the size of the data
    print("Data Size {}".format(mnist_train.data.shape))
    print("Num Classes {}".format(len(mnist_train.classes)))

    # Initialise the model and train
    model = MLP(28*28, 10)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-3
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 10
    
    losses = []
    accs = []

    for epoch in range(num_epochs):
        print("Epoch {} / {}".format(epoch, num_epochs))
        running_loss = 0.0
        for (data, target) in tqdm(train_loaders):
            # Each epoch the data will be shuffled, iterate over data
            """
            print("Batch:", batch_idx)
            print("Data shape:", data.shape)
            print("Target shape:", target.shape)
            """
            # Loop over the batch
            x = data.reshape(-1, 28*28).to(device)
            y_pred = model(x)

            # Estimate the loss
            loss = loss_function(y_pred, target.to(device))

            # Zero the param gradient
            optimiser.zero_grad()

            # Backpropaagate to compute gradients
            loss.backward()

            optimiser.step()
            # Run over validation set
            running_loss += loss.item()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_data in val_loader:
                images, labels = val_data
                images = images.reshape(-1, 28*28)
                outputs = model(images.to(device))
                _, predictions = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels.to(device)).sum().item()
        losses.append(running_loss)
        accs.append(100 * correct // total)
        print("Running Loss: {}".format(running_loss))
        print("Accuracy: {}%".format(100 * correct // total))
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(losses)
    ax[0].set_title("Running Loss over Epochs")
    ax[0].set(xlabel="Epoch #", ylabel="Running Loss")
    ax[1].plot(accs)
    ax[1].set_title("Accuracy over Epochs")
    ax[1].set(xlabel="Epoch #", ylabel="Accuracy %")
    plt.show()

    # Evaluate accuracy on the test data
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        print("TESTING MODEL")
        for test_data in test_loaders:
            images, labels = test_data
            images = images.reshape(-1, 28*28)
            outputs = model(images.to(device))
            _, predictions = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predictions == labels.to(device)).sum().item()
    print("Test Accuracy: {}%".format( 100* correct_test // total_test))





if __name__ == "__main__":
    print(torch.__version__)
    print("Cuda Available : {}".format(torch.cuda.is_available()))
    print("GPU - {0}".format(torch.cuda.get_device_name()))
    print("Running week 7")

    question3()
