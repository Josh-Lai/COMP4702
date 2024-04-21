#! /usr/bin/env python

import torchvision.transforms as transforms
import torchvision
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
def display_dataset():
  # Display the image
  dataiter = iter(trainloader)
  images, labels = next(dataiter)
  
  imshow(torchvision.utils.make_grid(images))
  print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class Net(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = torch.flatten(x, 1) # flatten all dimensions except batch
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x


def create_model(batch_size = 4):
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ]
  )
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

  net = Net()
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      #print(inputs)
      #print(labels)
      
      optimizer.zero_grad()
      outputs = net(inputs)
      # Still uncertain abt the  outputs of this CNNo
      #print(labels)
      #print(outputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

  PATH = "./cifar_net.pth"
  torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
  batch_size = 4
  #create_model(batch_size)
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ]
  )
  net = Net()
  net.load_state_dict(torch.load("./cifar_net.pth"))
  
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  # Test model on the test data
  dataiter = iter(testloader)
  images, labels = next(dataiter)

  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

  correct = 0
  total = 0

  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)

      # Choose the class with the highest energy
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
  
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predictions = torch.max(outputs, 1)
      
      # Collect the correct predictions for each class
      for label, prediction in zip(labels, predictions):
        if label == prediction:
          correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1
  
  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

