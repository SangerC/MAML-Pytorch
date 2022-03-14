#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.utils.prune as prune
import torch.nn.functional as F


# In[3]:


class GrowingNet(nn.Module):
    def __init__(self,n_output):
        super().__init__()
        conv1 = nn.Conv2d(3, 32, 3,padding_mode='zeros',padding=1)
        conv2 = nn.Conv2d(32, 64, 3,padding_mode='zeros',padding=1)
        conv3 = nn.Conv2d(64, 128, 3,padding_mode='zeros',padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layers = nn.ModuleList([conv1,conv2,conv3])
        self.fc1 = nn.Linear(256 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, n_output)

    def forward(self, x):
        pool = 1
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            
            ## Add 3 poolings in total
            num_pooling = 3
            if ((i+1) / len(self.conv_layers) * num_pooling) >= pool:
                x = self.pool(x)
                pool = pool + 1

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], 120, device = device) # check shape matching
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def add_conv(self, out_channel, disruption = False):
        
        zeros = id_matrix = [[0,0,0],[0,0,0],[0,0,0]]
        id_matrix = [[0,0,0],[0,1,0],[0,0,0]]
        layer = nn.Conv2d(self.conv_layers[-1].out_channels, out_channel, 
                                          3,padding_mode='zeros',padding=1, device = device)
        if not disruption:
            nn_weights = layer.weight.cpu().detach().numpy()

            for (index, i) in enumerate(nn_weights):
                for j in range(len(i)):
                    if index == j:
                        i[j] = id_matrix
                    else:
                        i[j] = zeros
            bias = torch.zeros(out_channel,device=device)
            layer.weight = torch.nn.Parameter(torch.Tensor(nn_weights).to(device),requires_grad=True)
            layer.bias = torch.nn.Parameter(bias)
        
        self.conv_layers.append(layer)
        
    def prune(self, amount = 0.25):
        parameters_to_prune = [(conv,'weight') for conv in net.conv_layers[conv_count:]]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        

