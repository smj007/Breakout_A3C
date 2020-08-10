#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 02:52:26 2020

@author: smj
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 

def normalized_column_initializer(weights, std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))  #in this stage, we get var(out) = std^2
    return out


def weights_init(m): #m here refers to the given network
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        in_size = np.prod(weight_shape[1:4]) #dim1*dim2*dim3
        out_size = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bounds = np.sqrt(6. / in_size + out_size)
        m.weight.data.uniform_(-w_bounds, w_bounds)
        m.bias.data.fill_(0)
        
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        in_size = weight_shape[1]
        out_size = weight_shape[0]
        w_bounds = np.sqrt(6. / in_size + out_size)
        m.weight.data.uniform_(-w_bounds, w_bounds)
        m.bias.data.fill_(0)
        
        
        
#brain of the A3C algorithm

class ActorCritic(nn.Module):
    
    def __init__(self, num_inputs, action_space):
        
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.lstm = nn.LSTMCell(32*32*3, 256) #this gives an op encoding of size 256, it helps to preserve a long term memory
        #each and every action such as the ball bouncing, etc, in a previous timestep is encoded and then it is retained thru LSTM
        num_outputs = action_space.n
        self.critic = nn.Linear(256, 1) #only V(s)
        self.actor = nn.Linear(256, num_outputs) #output Q(s, a)
        self.apply(weights_init)  #weights init is applied to the object itself (network)
        self.actor.weight.data = normalized_column_initializer(self.actor.weight.data, std = 0.01)
        self.critic.weight.data = normalized_column_initializer(self.critic.weight.data, std = 1.0) #the values of std are set low for A and high for V to set the exploration vs exploitation
        self.actor.bias.data.fill_(0)
        self.critic.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()  #this method is called to enable the dropout and BN wherever it is there


    def forward(self, inputs):
        
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*32*3)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx 
        return self.critic(x), self.actor(x), (hx, cx) #the last pair is used for the following future timesteps of the LSTM
    
    
            