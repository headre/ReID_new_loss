from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

class loss_WX_Y(nn.Module):
    def __init__(self):
        super(loss_WX_Y,self).__init__()

    def cal_W(source,target):
        ##需要W计算方式
        W=None
        return W

    def forward(self,source,target):
        W = self.cal_W(source,target)
        temp = W*source-target
        loss = temp.t()*temp
        return loss
