#! /usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super(MLP, self).__init__()
        # Create the layers of the model here
        # Output 10x1
        # Input 28x28

        # Set up the first layer
        self.fc1 = nn.Linear(dim_in, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, dim_out)
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
    train_subset, val_subset = random_split(
        mnist_train, [0.7, 0.3], generator=torch.Generator().manual_seed(1))
    # Split the train dataset for the validation daaset
    # Tensor equivalent of the data

    # Data loaders
    batch_size = 256
    train_loaders = DataLoader(
        train_subset, shuffle=True, batch_size=batch_size)
    test_loaders = DataLoader(mnist_test, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
    # Get the size of the data
    print("Data Size {}".format(mnist_train.data.shape))
    print("Num Classes {}".format(len(mnist_train.classes)))

    # Initialise the model and train
    model = MLP(28*28, 10)

    loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
    learning_rate = 1e-3
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    num_epochs = 10

    for epoch in range(num_epochs):
        print("Epoch {} / {}".format(epoch, num_epochs))
        for batch_idx, (data, target) in enumerate(train_loaders):
            # Each epoch the data will be shuffled, iterate over data
            """
            print("Batch:", batch_idx)
            print("Data shape:", data.shape)
            print("Target shape:", target.shape)
            """
            # Loop over the batch
            for idx in range(batch_size):
                x = data[idx, :].reshape(-1, 28*28)
                y_pred = model(x)

                # Estimate the loss
                print(data[idx, :].shape)
                print(target.shape)
                print(y_pred)
                print(target[0])
                loss = loss_function(y_pred, target[idx])

                # Zero the param gradient
                optimiser.zero_grad()

                # Backpropaagate to compute gradients
                loss.backward()

                optimiser.step()


if __name__ == "__main__":
    print(torch.__version__)
    print("Cuda Available : {}".format(torch.cuda.is_available()))
    print("GPU - {0}".format(torch.cuda.get_device_name()))
    print("Running week 7")

    question3()
