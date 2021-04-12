# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 12:08:02 2021

@author: LENOVO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('C:/Users/LENOVO/Desktop/data/sobar-72.csv')

data = data.apply(pd.to_numeric)
data_v = data.values
data_v = shuffle(data_v)

X = data_v[:, :19]
Y = data_v[:, 19]

xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, test_size = 0.3, random_state =42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(19, 38)
        self.fc2 = nn.Linear(38, 38)
        self.fc3 = nn.Linear(38, 19)
        self.fc4 = nn.Linear(19, 5)
        self.fc5 = nn.Linear(5,2)
        
    def forward(self, x):
        x = F.softmax(self.fc1(x))
        x = F.softmax(self.fc2(x))
        x = F.softmax(self.fc3(x))
        x = F.softmax(self.fc4(x))
        x = self.fc5(x)
        return x
net = Net()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.004)

epoch = 6000
xtrain = torch.Tensor(xtrain).float()
ytrain = torch.Tensor(ytrain).long()
xvalid = torch.Tensor(xvalid).float()
yvalid = torch.Tensor(yvalid).long()

for e in range(epoch):
    optimizer.zero_grad()
    y_pred = net(xtrain)
    loss1 = loss(y_pred, ytrain)
    loss1.backward()
    optimizer.step()
    if e % 100 == 0:
        print(e+1, epoch, loss1.item())
        out = net(xvalid)
        _, ypredv = torch.max(out.data, 1)
        
        print( metrics.accuracy_score(yvalid, ypredv))

out = net(xvalid)
_, ypredv = torch.max(out.data, 1)       
print(metrics.classification_report(yvalid, ypredv))
print(metrics.roc_auc_score (yvalid, ypredv))        
        
torch.save(net, 'C:/Users/LENOVO/Desktop/data/model.pt') 
        
        
        
        
        
        
        
        
        
        
        