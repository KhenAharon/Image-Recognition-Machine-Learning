import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# Loading datasets and defining loaders.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
my_dim = 128 * 5 * 5

indices = list(range(len(trainset)))
split = int(20 / 100 * len(indices))
val_i = np.random.choice(indices, size=split, replace=False)
train_i = list(set(indices) - set(val_i))

train_batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          sampler=SubsetRandomSampler(train_i), num_workers=4)
validloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          sampler=SubsetRandomSampler(val_i), num_workers=4)
test_batch_size = 1
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

train_size = len(train_i)  # 40000
valid_size = split  # 10000
test_size = len(testloader.dataset)  # 10000
# Defining the cnn.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input channel = 3 = RGB, output channel = 6 = kernels (conv depth), kernel size = 5 = filter size
        self.conv1 = nn.Conv2d(3, 48, 5)  # RGB, depth, filer size
        self.pool = nn.MaxPool2d(2, 2)  # window size, stride
        self.conv2 = nn.Conv2d(48, 128, 5)
        self.fc1 = nn.Linear(my_dim, 120)  # first dim was my dim
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.bn1 = nn.BatchNorm1d(120)
        self.bn2 = nn.BatchNorm1d(84)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # output -> 48*28*28
        # maxpooling -> 48 * 14 * 14
        x = self.pool(F.relu(self.conv2(x)))
        # output -> 128 * 10 * 10
        # maxpooling -> 128 * 5 * 5
        x = x.view(-1, my_dim)  # my_dim is a quarter
        x = F.relu(self.dropout1(self.bn1(self.fc1(x))))
        x = F.relu(self.dropout2(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


def train(net, optimizer, criterion):
    print("Training..")
    running_loss = 0.0
    losses1 = []
    correct = 0
    for idx, (data, labels) in enumerate(trainloader):
        inputs, labels = data.cuda(async=True), labels.cuda(async=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses1.append(loss.item())
        # print statistics
        running_loss += loss.item()
        if idx % 625 == 624:    # print every 625 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 625))
            running_loss = 0.0
    print("hits: {}".format(correct))
    losses1_avg.append(sum(losses1) / len(losses1))


def validation(net, criterion):
    print("Validating..")
    running_loss = 0.0
    losses2 = []
    correct = 0
    for idx, (data, labels) in enumerate(validloader):
        inputs, labels = data.cuda(async=True), labels.cuda(async=True)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        loss = criterion(outputs, labels)
        losses2.append(loss.item())
        # print statistics
        running_loss += loss.item()
        if idx % 157 == 156:    # print every 10000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 157))
            running_loss = 0.0
    losses2_avg.append(sum(losses2) / len(losses2))
    print("hits: {}".format(correct))
    if correct > 7500:
        print("a good accuracy!")
        #test(net, criterion)


def test(net, criterion):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    pred_str = ""
    losses = []
    y_pred = []
    y_test = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(testloader):
            y_test.append(labels.item())
            images, labels = images.cuda(async=True), labels.cuda(async=True)
            outputs = net(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)  # there was also 'outputs' alone (no .data)
            y_pred.append(predicted)
            c = (predicted == labels).squeeze()
            for i in range(test_batch_size):  # labels[i], c[i] - not good for 0-dim tensor.
                label = labels.item()
                class_correct[label] += c.item()
                class_total[label] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if pred_str == "":
                pred_str = str(predicted.item())
            else:
                pred_str = pred_str + "\n" + str(predicted.item())
        print("Test Hits: {}".format(correct))
        if correct > 7557:
            print("Saving a better test.pred file! ({} hits)".format(correct))
            my_file = "test{}.pred".format(correct)
            o_file = open(my_file, 'w')
            o_file.write(pred_str)  # writing the test.pred file
            o_file.close()
    print('Test Accuracy: {}%'.format(100 * correct / total))
    for i in range(10):
        print('Test Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    print("Test Loss: {}".format(sum(losses) / len(losses)))
    return y_pred, y_test


# The following is all the commands to run the first model. change the epochs from 0 to x to apply.
losses1_avg = []
losses2_avg = []
epochs = 0

net1 = Net()
net1.cuda()

# Defining hyper parameters, optimizations, loss function.
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(net1.parameters(), lr=0.0005)  # momentum=0.9

for epoch in range(epochs):
    net1.train()
    train(net1, optimizer1, criterion1)
    net1.eval()
    validation(net1, criterion1)
    # for testing, uncomment the following (I already have good results).
    #net.eval()
    #test(net1, criterion1)


#------------------------------------------- PART 2 -------------------------------------------#

# Loading datasets and defining loaders.
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

indices = list(range(len(trainset)))
split = int(20 / 100 * len(indices))
val_i = np.random.choice(indices, size=split, replace=False)
train_i = list(set(indices) - set(val_i))

train_batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          sampler=SubsetRandomSampler(train_i), num_workers=4)
validloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                          sampler=SubsetRandomSampler(val_i), num_workers=4)
test_batch_size = 1
testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

train_size = len(train_i)  # 40000
valid_size = split  # 10000
test_size = len(testloader.dataset)  # 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)
model_conv.cuda()
model_conv = model_conv.to(device)
criterion2 = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

losses1_avg = []
losses2_avg = []
epochs = 1

for epoch in range(epochs):

    model_conv.train()
    train(model_conv, optimizer_conv, criterion2)

    model_conv.eval()
    validation(model_conv, criterion2)

model_conv.eval()
y_pred, y_test = test(model_conv, criterion2)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd

for ix in range(10):
    print(ix, confusion_matrix(y_test,y_pred)[ix].sum())
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Visualizing of confusion matrix of final model
df_cm = pd.DataFrame(cm, range(10), range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()

# Plotting first model results!
plt.plot(np.arange(1, epochs+1), losses1_avg, 'ro-', label='Train', linewidth=2)
plt.plot(np.arange(1, epochs+1), losses2_avg, 'go-', label='Validation', linewidth=2)
plt.legend()
plt.title("Loss on the Training Set and Validation Set")
plt.show()
