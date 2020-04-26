#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from collections import OrderedDict
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[ ]:


train = torch.from_numpy(X_train).float()
labels = torch.from_numpy(y_train).long()
test = torch.from_numpy(X_test).float()
test_labels = torch.from_numpy(y_test).long()

input_size = 784
hidden_sizes = [128, 100, 64]
output_size = 2

dropout = 0.0
weight_decay = 0.0
n_chunks = 700
learning_rate = 0.03
optimizer = 'SGD'


# In[ ]:


train_acc = []
test_acc = []

for epochs in np.arange(10, 60, 10):
    model = build_model(input_size, output_size, hidden_sizes, dropout = dropout)

    fit_model(model, train, labels, epochs = epochs, n_chunks = n_chunks, learning_rate = learning_rate, weight_decay = weight_decay, optimizer = 'SGD')
    accuracy_train, accuracy_test = evaluate_model(model, train, y_train, test, y_test)

    train_acc.append(accuracy_train)
    test_acc.append(accuracy_test)

