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
        # self.drop_path_prob = 0.0
        self.batch_size=batch_size
        self.input_size=input_size
        # self.num_layers= 2
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.batch_first=batch_first
        self.labels_size = labels_size
        self.dropout= nn.Dropout(0.8)
        self.embedding = torch.nn.Embedding(self.vocab_size,self.input_size)
        self.lstm=torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,dropout=0.8,batch_first=self.batch_first)
        #做两层全连接
        self.classifier = nn.Linear(self.hidden_size,self.labels_size)
        self.sigmoid = nn.Sigmoid()

        # 定义损失函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,input,label=None):
        # input_shape= input.size()
        input = self.embedding(input)
        #lstm的返回包含3部分：output hn cn 对于分类我们应用的是隐藏层的输出即hn
        #output的维度为<batch_size,seq_length,hidden_size>
        output,hidden = self.lstm(input)
        output = self.dropout(output)
        output = output.contiguous() #contiguous:返回连续存储空间的张量；
        output=self.classifier(output)
        output = torch.nn.functional.softmax(output,dim=2)
        # output =self.sigmoid(output)
        output = output[:,-1,:]
        loss = 0
        if label is not None:
            loss = self.criterion(output,label)
        return output,loss
