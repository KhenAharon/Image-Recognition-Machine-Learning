# This file contains four models: A, B, C, D.

import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Loading Fashion-MNIST
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms)
test_set = datasets.FashionMNIST('./data', train=False, transform=transforms)

indices = list(range(len(train_set)))  # list of 0 to 59999
split = int(20 / 100 * len(indices))  # 12000
val_i = np.random.choice(indices, size=split, replace=False)
train_i = list(set(indices) - set(val_i))

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_i))
val_loader = torch.utils.data.DataLoader(train_set, batch_size=1, sampler=SubsetRandomSampler(val_i))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

train_size = len(train_i)  # 48000
valid_size = split  # 12000
test_size = len(test_loader.dataset)  # 10000
# Fashion-MNIST was loaded.


# Model A, B, C. batch_size = 64.   MODEL D- CNN at the end of file.
class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=0)
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        #x = F.relu(self.fc0(x)) # Model A
        #x = F.relu(self.dropout(self.fc0(x))) # Model B
        x = F.relu(self.dropout(self.bn0(self.fc0(x)))) # Model C
        #x = F.relu(self.fc1(x)) # Model A
        #x = F.relu(self.dropout(self.fc1(x))) # Model B
        x = F.relu(self.dropout(self.bn1(self.fc1(x)))) # Model C
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # dim=1 added to remove the warning.


model = FirstNet(image_size=28*28)
# Model A, B, C - End

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train():
    model.train()  # or model.eval
    train_loss = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
        loss = F.nll_loss(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= (train_size / batch_size)
    return train_loss, correct, train_size, (100. * correct / train_size)

# Validating
def validation():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  #data[0] sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= valid_size
    return test_loss, correct, valid_size, (100. * correct / valid_size)
    # ret  avg-loss, hits, examplesNo, accuracy

# Testing
def test():
    model.eval()
    test_loss = 0
    correct = 0
    pred_str = ""
    for data, target in test_loader:
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  #data[0] sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if pred_str == "":
            pred_str = str(pred.item())
        else:
            pred_str = pred_str + "\n" + str(pred.item())
    test_loss /= test_size
    if correct > 9061:
        print("Saving a better test.pred file!")
        #torch.save(model.state_dict(), "last_params")
        print("hits: {}".format(correct))
        my_file = "test{}.pred".format(correct)
        o_file = open(my_file, 'w')
        o_file.write(pred_str)  # writing the test.pred file
        o_file.close()
    return test_loss, correct, test_size, (100. * correct / test_size)
    # ret  avg-loss, hits, examplesNo, accuracy


# Running all together. Models A-C.
training_loss_avg = []
validation_loss_avg = []
epoches = 11 # 11
for epoch in range(1, epoches):
    print("Epoch number {}-".format(epoch))

    avg_loss1, hits1, examplesNo1, accuracy1 = train()
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss1, hits1, examplesNo1, accuracy1))
    training_loss_avg.append(avg_loss1)


    avg_loss2, hits2, examplesNo2, accuracy2 = validation()
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss2, hits2, examplesNo2, accuracy2))
    validation_loss_avg.append(avg_loss2)


    avg_loss3, hits3, examplesNo3, accuracy3 = test()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss3, hits3, examplesNo3, accuracy3))

    if epoch == 10:
        print("\nTraining set accuracy: {}/{} ({:.0f}%)".format(hits1, examplesNo1, accuracy1),
              "Validation set accuracy: {}/{} ({:.0f}%)".format(hits2, examplesNo2, accuracy2),
              "Test set accuracy: {}/{} ({:.0f}%)".format(hits3, examplesNo3, accuracy3),
              "Average training set loss: {}".format(avg_loss1),
              "Average validation set loss: {}".format(avg_loss2),
              "Average test set loss: {}".format(avg_loss3),
              sep="\n")

plt.plot(np.arange(1, epoches), training_loss_avg, 'ro-', label='Train', linewidth=2)
plt.plot(np.arange(1, epoches), validation_loss_avg, 'go-', label='Validation', linewidth=2)
plt.legend()
plt.title("Average loss per epoch - validation and training")
#plt.show()  # This is a blocking command.

# -------------------------------------------- CNN!! -----------------------------------------------------#
# Model D - CNN. tried on batch_size = 100. validation on batch too.

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_i))
val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(val_i))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# CNN END

# Initializing
cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

#  TESTING! function definition.
def convolution_test():
    cnn.eval()
    correct = 0
    total = 0
    pred_str = ""
    for images, labels in test_loader:
        images = Variable(images.float())
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        if pred_str == "":
            pred_str = str(predicted.item())
        else:
            pred_str = pred_str + "\n" + str(predicted.item())
    print("test hits:{}".format(correct))
    if correct > 9061:
        print("Saving a better test.pred file! ({} hits)".format(correct))
        my_file = "conv{}.pred".format(correct)
        o_file = open(my_file, 'w')
        o_file.write(pred_str)  # writing the test.pred file
        o_file.close()

# TRAINING!

losses1_avg = []
losses2_avg = []

num_epochs = 10  # 10
train_dataset = train_set
for epoch in range(num_epochs):
    losses1 = []
    losses2 = []
    cnn.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        # added 3 lines
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses1.append(loss.item())
        if (i + 1) % batch_size == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, train_size // batch_size, loss.item()))
    print("Epoch {} train hits: {}".format(epoch+1, correct))
    print("Average loss for this epoch: {}".format(sum(losses1)/len(losses1)))
    losses1_avg.append(sum(losses1)/len(losses1))

    # VALIDATING!
    cnn.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(val_loader):
        images = Variable(images.float())
        labels = Variable(labels)
        outputs = cnn(images)
        # added 3 lines
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        losses2.append(loss.item())
        if (i + 1) % batch_size == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, valid_size // batch_size, loss.item()))
    print("Epoch {} Val hits: {}, val precision: {}".format(epoch+1, correct, correct/valid_size))
    print("Average loss for this epoch: {}".format(sum(losses1)/len(losses1)))
    losses2_avg.append(sum(losses2)/len(losses2))
    if correct > 10000:  # good validation, let's test with the same parameters!
        convolution_test()

# Plotting the CNN results!
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))
print("hits: {}".format(correct))
plt.plot(np.arange(1, num_epochs+1), losses1_avg, 'ro-', label='Train', linewidth=2)
plt.plot(np.arange(1, num_epochs+1), losses2_avg, 'go-', label='Validation', linewidth=2)
plt.legend()
plt.title("CNN!! Average loss per epoch - training and validation")
plt.show()

# torch.save(model.state_dict(), "last_params") - cmd to save best params, a comment for me.