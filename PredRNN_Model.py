import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from PredRNN_Cell import PredRNNCell
##############################################
#
# 构造PredRNN
# Construct PredRNN
#
##############################################
class PredRNN(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, hidden_dim_m, kernel_size, num_layers,
                 batch_first=False, bias=True):
        super(PredRNN, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers) # 按照层数来扩充 卷积核尺度/可自定义
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers) # 按照层数来扩充 LSTM单元隐藏层维度/可自定义
        hidden_dim_m = self._extend_for_multilayer(hidden_dim_m, num_layers)  # M的单元应保持每层输入和输出的一致性.
        if not len(kernel_size) == len(hidden_dim) == num_layers:  # 判断相应参数的长度是否与层数相同
            raise ValueError('Inconsistent list length.')
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_m = hidden_dim_m
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        cell_list = []
        for i in range(0, self.num_layers):
            if i == 0:    # 0 时刻， 图片的输入即目前实际输入
                cur_input_dim = self.input_dim
            else:
                cur_input_dim = self.hidden_dim[i - 1]  # 非0时刻，输入的维度为上一层的输出
                #Cell_list.appenda为堆叠层操作
            cell_list.append(PredRNNCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          hidden_dim_m=self.hidden_dim_m[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias).cuda())
        self.cell_list = nn.ModuleList(cell_list)#Cell_list进行Model化
    def forward(self, input_tensor, hidden_state=False, hidden_state_m=False):
        if self.batch_first is False:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        if hidden_state is not False:
            hidden_state = hidden_state
        else:   #如果没有输入自定义的权重，就以0元素来初始化
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        if hidden_state_m is False:
            h_m = Variable(torch.zeros(input_tensor.shape[0], self.hidden_dim_m[0],
                                       input_tensor.shape[3], input_tensor.shape[4])
                           ,requires_grad=True).cuda()
        else:
            h_m = hidden_state_m

        layer_output_list = []  #记录输出
        layer_output_list_m = []  # 记录每一层m
        layer_output_list_c = []  # 记录每一层c
        last_state_list = []  #记录最后一个状态
        layer_output_list_m = []  # 记录最后一个m
        last_state_list_m = []  # 记录最后一个m
        seq_len = input_tensor.size(1)   #第二个时间序列，3
        cur_layer_input_1 = input_tensor #x方向上的输入
        all_layer_out = []
        for t in range(seq_len):
            concat=[]
            output_inner_c = []  # 记录输出的c
            output_inner = []  #记录输出的c
            output_inner_m = []  # 记录输出的m
            output_inner_h_c=[] # 记录输出的h 和c
            h0 = cur_layer_input_1[:, t, :, :, :]  #确定layer = 1 时的输入,如雷达回波图等矩阵信息
            for layer_idx in range(self.num_layers): # 由于M在layer上传递,所以优先考虑layer上的传递
                if t == 0:  # 由于M在layer上传递，所以要区分t=0(此时m初始化)
                    h, c = hidden_state[layer_idx]  # h和c来自于初始化/自定义
                    h, c, h_m = self.cell_list[layer_idx](input_tensor=h0,
                                                          cur_state=[h, c], cur_state_m=h_m)#经过一个cell/units输出的h,c,m
                    output_inner_c.append(c) #记录输出的c进行
                    output_inner.append(h)
                    output_inner_m.append(h_m)
                    output_inner_h_c.append([h,c])
                    h0=h
                else:
                    h = cur_layer_input[layer_idx]
                    c = cur_layer_input_c[layer_idx]
                    h, c, h_m = self.cell_list[layer_idx](input_tensor=h0,
                                                          cur_state=[h, c], cur_state_m=h_m)
                    output_inner_c.append(c)
                    output_inner.append(h)
                    output_inner_m.append(h_m)
                    output_inner_h_c.append([h, c])
                    h0 = h
            cur_layer_input = output_inner#记录某个t，全部layer的输出h
            cur_layer_input_c = output_inner_c#记录某个t，全部layer的输出c
            cur_layer_input_m = output_inner_m#记录某个t，全部layer的输出m
            alllayer_output = torch.cat(output_inner, dim=1) #把某个t时刻每个隐藏层的输出进行堆叠,以便于在解码层参照Convlstm使用1x1卷积得到输出
            all_layer_out.append(alllayer_output)#记录每个t时刻,所有隐藏层输出的h,以便于在解码层参照Convlstm使用1x1卷积得到输出
            per_time_all_layer_stack_out=torch.stack(all_layer_out, dim=1)#记录每个t时刻,所有隐藏层输出的h,以便于在解码层参照Convlstm使用1x1卷积得到输出
            layer_output_list.append(h)# 记录每一个t得到的最后layer的输出h
            last_state_list.append([h, c])#记录每一个t得到的最后layer的输出h,C
            last_state_list_m.append(h_m)#记录每一个t得到的最后layer的输出m
            #按层对最后一层的H和C进行扩展
            # ↓↓↓↓↓↓↓↓↓全部t时刻最后layer的输出h
            # ↓↓↓↓↓↓↓↓↓最后t时刻全部layer的输出h和c
            # ↓↓↓↓↓↓↓↓↓全部t时刻最后layer的输出m/t+1时刻0 layer的输入m
            # ↓↓↓↓↓↓↓↓↓全部时刻全部layer的h在隐藏层维度上的总和，hidden_dim = [7,1],则输出channels = 8
            return torch.stack(layer_output_list, dim=1),\
               output_inner_h_c,\
               torch.stack(last_state_list_m, dim=0),\
               per_time_all_layer_stack_out

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
    @staticmethod

    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
    @staticmethod

    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param