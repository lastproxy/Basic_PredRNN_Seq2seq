import torch.nn as nn
from torch.autograd import Variable
import torch
####################################
#
# 单层，单时间步的PredRNNCell(细胞/单元)，用于构造整个外推模型
# The cell/unit of predrnncell of every layer and time_step, for constructing the entire extrapolation model.
#
####################################
class PredRNNCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim_m, hidden_dim,kernel_size, bias):
        super(PredRNNCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_m = hidden_dim_m   #  hidden of M
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        #####################################################################################
        # 相应符号可对应参照论文
        # Corresponding symbols can correspond to reference paper
        # conv_h_c for gt, it, ft
        # conv_m for gt', it', ft'
        # conv_o for ot
        # self.conv_h_next for Ht
        self.conv_h_c = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_m = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim_m,
                              out_channels=3 * self.hidden_dim_m,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim * 2 + self.hidden_dim_m,
                              out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.conv_h_next = nn.Conv2d(in_channels=self.hidden_dim + self.hidden_dim_m,
                                out_channels=self.hidden_dim,
                                kernel_size=1,
                                bias=self.bias)
    def forward(self, input_tensor, cur_state, cur_state_m):
        h_cur, c_cur= cur_state  #cur = Current input of H and C
        h_cur_m = cur_state_m #cur = Current input of m

        combined_h_c = torch.cat([input_tensor,h_cur], dim=1)
        combined_h_c = self.conv_h_c(combined_h_c)
        cc_i, cc_f, cc_g = torch.split(combined_h_c, self.hidden_dim, dim=1)

        combined_m = torch.cat([input_tensor,  h_cur_m], dim=1)
        combined_m = self.conv_m(combined_m)
        cc_i_m, cc_f_m, cc_g_m = torch.split(combined_m, self.hidden_dim_m, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        i_m = torch.sigmoid(cc_i_m)
        f_m = torch.sigmoid(cc_f_m)
        g_m = torch.tanh(cc_g_m)
        h_next_m = f_m * h_cur_m + i_m * g_m

        combined_o = torch.cat([input_tensor, h_cur, c_next, h_next_m], dim=1)
        combined_o = self.conv_o(combined_o)
        o = torch.sigmoid(combined_o)

        h_next = torch.cat([c_next, h_next_m], dim=1)
        h_next = self.conv_h_next(h_next)
        h_next = o * torch.tanh(h_next)

        return h_next, c_next, h_next_m
    #####################################
    #
    # 用于在t=0时刻时初始化H,C,M
    # For initializing H,C,M at t=0
    #
    #####################################
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())