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
IMAGE_PATH = 'F:/张智察/数据转移/Python/program/Keras/ImagesSequencesPredictions-master/samples/'
WIDTH = 100
HEIGHT = 100
FRAMES = 49
channels=1
SEQUENCE = np.array([])
BASIC_SEQUENCE = np.array([])
NEXT_SEQUENCE = np.array([])
NUMBER = 0
for file in os.listdir(IMAGE_PATH):
    image=os.path.join(IMAGE_PATH, file) #读取文件路径
    picture = Image.open(image)  #打开当前图片
    picture = picture.crop((1920//4,1440//3, 1920*6//10, 1440*7//10))  #进行截取
    picture = picture.resize((WIDTH, HEIGHT), Image.ANTIALIAS) #重新定义像素，转化为黑白图
    picture = picture.convert('L')  #进行转化为L图,L图意味单通道，因此下面第二行channels为1
    picture.save('C:/Users/yaoyuye/Desktop/samples/'+file)  # 批量保存
    data = np.array(picture.getdata()).reshape(1,WIDTH, HEIGHT) #定义每个图片的像素大小，并且读取进去数据
    image_array = data  #暂存当前图片资料
    SEQUENCE = np.append(SEQUENCE, image_array)  #每个图片资料进行拼接,最终集合成全部图像数据，
    NUMBER += 1
    print(NUMBER)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
SEQUENCE = SEQUENCE.reshape(NUMBER, channels*WIDTH * HEIGHT)  ##弄成80张图片，所有序列一起排列.
np.savez('sequence_array.npz', sequence_array=SEQUENCE)
SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array
X_train = SEQUENCE.reshape(NUMBER, channels,WIDTH , HEIGHT)

X_train_max=X_train.max()
X_train_min=X_train.min()
X_train = (X_train - X_train_min)/(X_train_max-X_train_min)
X_train_1 = np.zeros((NUMBER-FRAMES, FRAMES,1,WIDTH, HEIGHT))   #64张图，16帧，
X_train_2 = np.zeros((NUMBER-FRAMES,  FRAMES,1,WIDTH, HEIGHT))    #64张图，16针，在BASIC的下一个时次
for i in range(FRAMES):
    print(i)
    X_train_1[ :, i,:, :, :] = X_train[i:i+NUMBER-FRAMES]   #把原始雷达图中的1-64,2-65,.....,64-80给BASIC中的每一针
    X_train_2[:,i, :, :, :] = X_train[i+1:i+NUMBER-FRAMES+1] #把BASIC的往后推一个时次,
a=torch.tensor(X_train_1)
a.shape
a=a.float()
b=torch.tensor(X_train_2)
b=b.float()
input=a[0:1,:10,:,:,:].cuda()
target=b[0:1,:10,:,:,:].cuda()
class PredRNN_enc(nn.Module):
    def __init__(self):
        super(PredRNN_enc, self).__init__()
        self.pred1_enc=PredRNN(input_size=(100,100),
                input_dim=1,
                hidden_dim=[7, 1],
                hidden_dim_m=[7, 7],
                kernel_size=(11, 11),
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
                kernel_size=(11, 11),
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
    enc_hidden, enc_h_y = enc(input)
    for i in range(input.shape[1]):
        optimizer.zero_grad()
        out,layer_h_c,last_h_y = dec(input[:,i:i+1,:,:,:], enc_hidden, enc_h_y[-1])
        loss=loss_fn(out,target[:,i:i+1,:,:,:])
        loss_total+=loss
        enc_hidden = layer_h_c
        enc_h_y = last_h_y
    loss_total=loss_total/input.shape[1]
    loss_total.backward()
    optimizer.step()
    print(epoch,epoch,loss_total)


input_test = a.cuda()
target_test = b.cuda()
enc_hidden_test, enc_h_y_test = enc(a[0:1,:10,:,:,:].cuda())
out_test = input_test[:, 0:0 + 1, :, :, :]
loss_total_test = 0
for i in range(input_test.shape[1]):
    out_test, layer_h_c_test, last_h_y_test = dec(out_test, enc_hidden_test, enc_h_y_test[-1])
    loss_test = loss_fn(out_test, target_test[:, i:i + 1, :, :, :])
    print(loss_test)
    loss_total_test += loss_test
    enc_hidden_test = layer_h_c_test
    enc_h_y_test = last_h_y_test
    out = out_test.cpu()
    out = out.detach().numpy()
    plt.axis('off')
    plt.imshow(out[0, 0, 0, :, :],cmap='binary_r')
    plt.savefig('C:/Users/yaoyuye/Desktop/pytorch_work/PRED/' + str(i))
print(loss_total_test/input_test.shape[1])


for i in range(input_test.shape[1]):
    plt.axis('off')
    plt.imshow(b[0, i, 0, :, :],cmap='binary_r')
    plt.savefig('C:/Users/yaoyuye/Desktop/pytorch_work/ACTUAL/' + str(i))