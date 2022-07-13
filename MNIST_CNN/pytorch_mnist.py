"""
adapted from: https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
changes:
    - tensor.item() for 0-dim tensors
    - batch-size to 128
    - model architecture and optimizer adapted to keras example
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

batch_size = 128
num_classes = 10
epochs = 12

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./MNIST_CNN/mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              # transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=batch_size, 
                                           shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./MNIST_CNN/mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              # transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=batch_size, 
                                           shuffle=True)

class CNNClassifier(nn.Module):
    """Adapted model from tensorflow example"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d(2)
        self.droupout1 = nn.Dropout(0.25)
        self.dense1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.droupout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x))
        
        return x

# create classifier and optimizer objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = CNNClassifier().to(device)
opt = optim.Adadelta(clf.parameters(), lr=0.1, rho=0.95, eps=1e-07)

loss_history = []
acc_history = []

def train(epoch):
    clf.train() # set model in training mode (need this because of dropout)
    
    # dataset API gives us pythonic batching 
    for batch_id, (data, label) in enumerate(train_loader):
        data = Variable(data).to(device)
        target = Variable(label).to(device)
        
        # forward pass, calculate loss and backprop!
        opt.zero_grad()
        preds = clf(data)
        loss = F.nll_loss(preds, target)
        loss.backward()
        opt.step()
        
        # if batch_id % 100 == 0:
        #     print(loss.item())

def test(epoch):
    clf.eval() # set model in inference mode (need this because of dropout)
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            
            output = clf(data)
            test_loss += F.nll_loss(output, target).item()
            
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(test_loader) # loss function already averages over batch size
        test_accuracy = correct / len(test_loader.dataset)
        acc_history.append(test_accuracy)
        loss_history.append(test_loss)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     test_accuracy))

for epoch in range(epochs):
    # print("Epoch %d" % epoch)
    train(epoch)
    test(epoch)

with open('log_pytorch.csv', 'a') as f:
    # f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")
    for epoch in range(epochs):
        f.write("{},0,0,{},{}\n".format(epoch, acc_history[epoch], loss_history[epoch]))