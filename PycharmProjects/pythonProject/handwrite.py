# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

# Even if you don't use them, you still have to import
import random
import numpy as np


# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=784, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


# Train
def train(device, model, epochs, optimizer, loss_function, train_loader):
    for epoch in range(1, epochs+1):
        for times, data in enumerate(train_loader, 1):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward and backward propagation
            outputs = model(inputs.view(inputs.shape[0], -1))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Show progress
            if times % 100 == 0 or times == len(train_loader):
                print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, epochs, times, len(train_loader), loss.item()))

    return model


def test(device, model, test_loader):
    # Settings
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs.view(inputs.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accurecy:', correct / total)


def main():
    # GPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device state:', device)

    # Settings
    epochs = 10
    batch_size = 50
    lr = 0.001
    loss_function = nn.NLLLoss()
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Transform
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    # Data
    train_set = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
    test_set = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Train
    model = train(device, model, epochs, optimizer, loss_function, train_loader)

    # Test
    test(device, model, test_loader)


if __name__ == '__main__':
    main()