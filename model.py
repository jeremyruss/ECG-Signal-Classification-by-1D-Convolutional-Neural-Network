import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import statistics

from sklearn.utils import shuffle
from sklearn.utils import resample
from random import randrange
from random import randint
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_train = pd.read_csv("mitbih_train.csv", header=None)
df_test = pd.read_csv("mitbih_test.csv", header=None)

balance = df_train[187].value_counts()
print(balance)

def upsample_data(dataframe):
    df_0 = dataframe[dataframe[187]==0].sample(n=30000)
    df_1 = dataframe[dataframe[187]==1]
    df_2 = dataframe[dataframe[187]==2]
    df_3 = dataframe[dataframe[187]==3]
    df_4 = dataframe[dataframe[187]==4]

    df_1_up = resample(df_1, replace=True, n_samples=5000)
    df_2_up = resample(df_2, replace=True, n_samples=10000)
    df_3_up = resample(df_3, replace=True, n_samples=2500)
    df_4_up = resample(df_4, replace=True, n_samples=10000)

    df = pd.concat([df_0, df_1_up, df_2_up, df_3_up, df_4_up])

    balance = df[187].value_counts()
    print(balance)
    return df

df_train = upsample_data(df_train)
    
x_train = torch.tensor(df_train.iloc[:,:186].values.tolist())
x_test = torch.tensor(df_test.iloc[:,:186].values.tolist())

def add_noise(tensor, std, mean):
    normal = torch.normal(mean, std, size=(len(x_train), 1, 186))
    output = tensor + normal
    return output

#x_train = add_noise(x_train, 0.05, 0)

y_train = torch.tensor(df_train[187].values, dtype=torch.long)
y_test = torch.tensor(df_test[187].values, dtype=torch.long)

x_train = x_train.view(-1, 1, 186)
x_test = x_test.view(-1, 1, 186)

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

train_len = int(len(x_train))
test_len = int(len(x_test))

# Shuffle, balance & add noise

def display(dataframe):
    rand = randrange(len(dataframe)-1)
    data = dataframe.iloc[rand,0:186]
    category = dataframe.iloc[rand, 187].astype(int)
    title = "Class: " + category.astype(str)
    plt.plot(data)
    plt.title(title)
    plt.show()

#display(df_train)

EPOCHS = 1
BATCH_SIZE = 100
CHUNK = 4000
LR = 0.001
PATH = 'model.pth'
VALIDATE = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(1984, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 5)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self. maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self. maxpool2(x)
        x = x.view(-1, 64*31)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

net.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

x_train, x_test = x_train.to(device), x_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

def acc_func(out, labels):
    _, predicted = torch.max(out, 1)
    total = len(labels)
    correct = (predicted == labels).sum().item()
    accuracy = correct/total
    return accuracy

def validation():
    val_accs = []
    val_loss = []
    for i in range(0, test_len, BATCH_SIZE):
        x_batch = x_test[i:i+BATCH_SIZE]
        y_batch = y_test[i:i+BATCH_SIZE]
        with torch.no_grad():
                outputs = net(x_batch)
        _val_loss = loss_func(outputs, y_batch)
        val_loss.append(_val_loss.item())
        val_acc = acc_func(outputs, y_batch)
        val_accs.append(val_acc)
    
    val_acc = statistics.mean(val_accs)
    val_loss = statistics.mean(val_loss)

    return val_acc, val_loss


training_loss = []
training_acc = []
validation_loss = []
validation_acc = []

def train():

    for epoch in range(EPOCHS):

        print('EPOCH_______________________________', epoch+1)

        loss = []
        accs = []

        for i in range(0, train_len, BATCH_SIZE):

            x_batch = x_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]

            net.zero_grad()

            outputs = net(x_batch)
            _loss = loss_func(outputs, y_batch)
            loss.append(_loss.item())

            _loss.backward()
            optimizer.step()

            acc = acc_func(outputs, y_batch)
            accs.append(acc)

            # Provides statistics on validation loss and accuracy, slower runtime
            
            if VALIDATE:
                val_acc, val_loss = validation()
            else:
                val_acc, val_loss = 0, 0

            if i % CHUNK == 0:
                accuracy = statistics.mean(accs)
                loss = statistics.mean(loss)

                training_loss.append(loss)
                training_acc.append(accuracy)
                validation_loss.append(val_loss)
                validation_acc.append(val_acc)

                print('[', i, '/', train_len, ']',
                    'Loss: ', round(float(loss), 4),
                    'Accuracy: ', round(float(accuracy), 3),
                    'Val Loss: ', round(float(val_loss), 4),
                    'Val Accuracy: ', round(float(val_acc), 3))
                loss = []
                accs = []

    print('Finished training!')

train()

# Generate learning curves

def learning_curves(loss, acc, val_loss, val_acc):
    fig, ax = plt.subplots(2)
    ax[0].plot(loss, label="loss")
    ax[0].plot(acc, label="acc")
    ax[0].legend(loc=2)
    ax[1].plot(val_loss, label="val_loss")
    ax[1].plot(val_acc, label="val_acc")
    ax[1].legend(loc=2)
    plt.show()

learning_curves(training_loss, training_acc, validation_loss, validation_acc)

# Save & load model

def save():
    torch.save(net.state_dict(), PATH)

def load():
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    return net

# Query the model

def query(dataframe):

    size = 20

    df_0 = dataframe[dataframe[187]==0]
    df_1 = dataframe[dataframe[187]==1]
    df_2 = dataframe[dataframe[187]==2]
    df_3 = dataframe[dataframe[187]==3]
    df_4 = dataframe[dataframe[187]==4]

    df_0 = df_0[:size]
    df_1 = df_1[:size]
    df_2 = df_2[:size]
    df_3 = df_3[:size]
    df_4 = df_4[:size]

    df = pd.concat([df_0, df_1, df_2, df_3, df_4])

    x = torch.tensor(df.iloc[:, :186].values.tolist())
    y = torch.tensor(df[187].values, dtype=torch.long)

    x = x.view(-1, 1, 186)

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        out = net(x)

    _, predicted = torch.max(out, 1)

    df = df.values.tolist()

    for i in range(len(df)):
        df[i].append(int(predicted[i]))

    return df
    
# Export data to JSON

def export_JSON():
    data = {'data': query(df_test),
            'acc': training_acc,
            'loss': training_loss,
            'val_acc': validation_acc,
            'val_loss': validation_loss}
    with open('data.json', 'w') as f:
        json.dump(data, f)

export_JSON()

print('Completed!')