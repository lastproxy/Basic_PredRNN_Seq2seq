from PredRNN_Model import PredRNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import time

input=torch.rand(1,1,1,100,100).cuda()   # Batch_size , time_step, channels, hight/width, width/hight
target=torch.rand(1,1,1,100,100).cuda()   # Batch_size , time_step, channels, hight/width, width/hight

class PredRNN_enc(nn.Module):
    def __init__(self):
        super(PredRNN_enc, self).__init__()
        self.pred1_enc=PredRNN(input_size=(100,100),
                input_dim=1,
                hidden_dim=[7, 1],
                hidden_dim_m=[7, 7],
                kernel_size=(7, 7),
                num_layers=2,
                batch_first=True,
                bias=True).cuda()
    def forward(self,enc_input):
        _, layer_h_c, all_time_h_m, _ = self.pred1_enc(enc_input)
        return layer_h_c, all_time_h_m
class PredRNN_dec(nn.Module):
    def __init__(self):
        super(PredRNN_dec, self).__init__()
        self.pred1_dec=PredRNN(input_size=(100,100),
                input_dim=1,
                hidden_dim=[7, 1],
                hidden_dim_m=[7, 7],
                kernel_size=(7, 7),
                num_layers=2,
                batch_first=True,
                bias=True).cuda()
        self.relu = nn.ReLU()
    def forward(self,dec_input,enc_hidden,enc_h_m):
        out, layer_h_c, last_h_m, _ = self.pred1_dec(dec_input,enc_hidden,enc_h_m)
        out = self.relu(out)
        return out, layer_h_c, last_h_m
enc=PredRNN_enc().cuda()
dec=PredRNN_dec().cuda()

import itertools
loss_fn=nn.MSELoss()
position=0
optimizer=optim.Adam(itertools.chain(enc.parameters(), dec.parameters()),lr=0.001)
for epoch in range(1000):
    loss_total=0
    enc_hidden, enc_h_m = enc(input)
    for i in range(input.shape[1]):
        optimizer.zero_grad()
        out, layer_h_c, last_h_m = dec(input[:,i:i+1,:,:,:], enc_hidden, enc_h_m[-1])
        loss=loss_fn(out, target[:,i:i+1,:,:,:])
        loss_total+=loss
        enc_hidden = layer_h_c
        enc_h_m = last_h_m
    loss_total=loss_total/input.shape[1]
    loss_total.backward()
    optimizer.step()
    print(epoch,epoch,loss_total)