import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper paramaters
n_epochs = 150
learning_rate = 0.0005
batch_size = 100

# dataset arrangement
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='Users/talgeller/Desktop/cifar 10', train=True, download=False, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='Users/talgeller/Desktop/cifar 10', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# module setting
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(128*5*5, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 128)
        self.linear8 = nn.Linear(128, 128)
        self.linear9 = nn.Linear(128, 128)
        self.linear10 = nn.Linear(128, 10)
        self.relu = F.relu

    def forward(self, x):
        outputs_conv = self.conv1(x)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.pool1(outputs_conv)
        outputs_conv = self.conv2(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.pool2(outputs_conv)
        outputs_conv = outputs_conv.view(-1, 128*5*5)
        outputs_conv = self.linear1(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear2(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear3(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear4(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear5(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear6(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear7(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear8(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear9(outputs_conv)
        outputs_conv = self.relu(outputs_conv)
        outputs_conv = self.linear10(outputs_conv)
        return outputs_conv


model = ConvNet().to(device)

# cost and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#optimize loop
for epoch in range(n_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        ypred = model(features)
        cost = criterion(ypred, labels)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

#test
with torch.no_grad():

    correct_pred = 0
    num_pred = 0
    for i, (features, labels) in enumerate(test_loader):
        outputs = model(features)
        num_pred += outputs.size(0)
        _,pred = torch.max(outputs, 1)
        correct_pred += (pred == labels).sum().item()


print(correct_pred/num_pred)
