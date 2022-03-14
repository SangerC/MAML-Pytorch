#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.utils.prune as prune
import torch.nn.functional as F


# In[3]:


class GrowingNet(nn.Module):

    def __init__(self, config, imgc, imgsz):
        super().__init__()

        self.config = config

        self.vars = nn.ModuleList()
        self.vars_bn = nn.ModuleList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                self.vars.append(nn.Conv2d(param[1], param[0], param[2], padding_mode='zeros', padding=1))

            elif name is 'linear':
                self.vars.append(nn.Linear(param[1], param[0]))

            elif name is 'bn':
                self.vars.append(nn.BatchNorm2d(param[0]))
                self.vars_bn.append(nn.BatchNorm2d(param[0], requires_grad=False))
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def forward(self, x, vars=None, bn_training=True):
        if vars = None:
            vars = self.vars

        idx = 0
        bn_inx = 0

        for name, param in self.config:
            if name is 'conv2d':
                x = vars[idx](x)
                idx += 1
            elif name is 'linear':
                x = vars[idx](x)
                idx += 1
            elif name is 'bn':
                #################
            elif is 'flatten':
                x = x.view(x.size(0), -1)
            elif is 'reshape':
                x = x.view(x.size(0), -1)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
   
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
        

