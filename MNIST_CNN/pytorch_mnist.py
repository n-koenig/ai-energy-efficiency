"""
adapted from: https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
changes:
    - tensor.item() for 0-dim tensors
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
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              # transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=batch_size, 
                                           shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              # transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                          ])), 
                                           batch_size=batch_size, 
                                           shuffle=True)

class CNNClassifier1(nn.Module):
    """Custom module for a simple convnet classifier"""
    def __init__(self):
        super(CNNClassifier1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # input is 28x28x1
        # conv1(kernel=5, filters=10) 28x28x10 -> 24x24x10
        # max_pool(kernel=2) 24x24x10 -> 12x12x10
        
        # Do not be afraid of F's - those are just functional wrappers for modules form nn package
        # Please, see for yourself - http://pytorch.org/docs/_modules/torch/nn/functional.html
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # conv2(kernel=5, filters=20) 12x12x20 -> 8x8x20
        # max_pool(kernel=2) 8x8x20 -> 4x4x20
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        
        # flatten 4x4x20 = 320
        x = x.view(-1, 320)
        
        # 320 -> 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        # 50 -> 10
        x = self.fc2(x)
        
        # transform to logits
        return F.log_softmax(x)

class CNNClassifier2(nn.Module):
    """Adapted model from tensorflow example"""
    def __init__(self):
        super(CNNClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d(2)
        self.droupout1 = nn.Dropout(0,25)
        self.dense1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0,5)
        self.dense2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.droupout1(x)
        x = x.reshape(-1, 64*12*12)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x))
        
        return x

# create classifier and optimizer objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = CNNClassifier2().to(device)
opt = optim.Adadelta(clf.parameters(), lr=0.1)

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
        loss_history.append(loss.item())
        opt.step()
        
        # if batch_id % 100 == 0:
        #     print(loss.item())

    return loss.item()

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

        test_loss = test_loss
        test_loss /= len(test_loader) # loss function already averages over batch size
        accuracy = 100. * correct / len(test_loader.dataset)
        acc_history.append(accuracy)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     accuracy))

    return accuracy, test_loss

# with open("log_pytorch", 'a') as f:
#     f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")

for epoch in range(epochs):
    loss = train(epoch)
    test_accuracy, test_loss = test(epoch)
    with open('log_pytorch_test.csv', 'a') as f:
        f.write("{},0,{},{},{}\n".format(epoch, loss, test_accuracy, test_loss))