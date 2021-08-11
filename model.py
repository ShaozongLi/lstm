# -*- coding:utf-8 -*-
# @Time: 2021/3/16 11:29
# @Author: ShaozongLi
# @Email: lishaozong@hikvision.com.cn
# @File: model.py
import torch
from torch.nn import Module
import torch.nn as nn


class LstmModel(Module):
    def __init__(self,batch_size,input_size,hidden_size,vocab_size,labels_size,batch_first=True):
        super(LstmModel,self).__init__()
        self.drop_path_prob = 0.0
        self.batch_size=batch_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.batch_first=batch_first
        self.dropout=0.5
        self.embedding = torch.nn.Embedding(self.vocab_size+1,self.input_size)
        self.lstm=torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,dropout=self.dropout,batch_first=self.batch_first)
        #做一层全连接
        self.classifier = nn.Linear(self.hidden_size,labels_size)

        # 定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,input,label=None):
        input_shape= input.size()
        input = self.embedding(input)
        #lstm的返回包含3部分：output hn cn 对于分类我们应用的是隐藏层的输出即hn
        output,hidden = self.lstm(input,)
        output = hidden[0].view(input_shape[0],-1)
        output=self.classifier(output)
        #做softmax归一化
        output = nn.functional.softmax(output,dim=1)
        loss = 0
        if label is not None:
            loss = self.criterion(output,label)
            # print(f"loss is {loss}")
        return output,loss
