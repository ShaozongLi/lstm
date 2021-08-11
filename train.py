# -*- coding:utf-8 -*-
# @Time: 2021/3/16 11:39
# @Author: ShaozongLi
# @Email: lishaozong@hikvision.com.cn
# @File: train.py
import torch

from lstm.data_set import LstmDataSet
from torch.utils.data.dataloader import DataLoader

from lstm.model import LstmModel
from lstm.tokenizer import LstmTokenzier


def init_hidden(batch_size,hidden_size,device):
    '''
    初始化h0和c0
    h0和c0的维度为（num_layer,batch_size,hidden_size）
    :return:
    '''
    return torch.zeros(1, batch_size, hidden_size).to(device), torch.zeros(1, batch_size, hidden_size).to(device)


if __name__== "__main__":
    EPOACH=20
    BATCH_SIZE = 128
    INPUT_SIZE = 512
    HIDDEN_SIZE = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LstmTokenzier()
    train_data = LstmDataSet("./data/train.tsv")
    train_dataloader = DataLoader(dataset=train_data,batch_size=64,shuffle=True)
    #model
    model = LstmModel(batch_size=BATCH_SIZE, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=len(
        tokenizer.vocab),labels_size=2).to(device)
    model = torch.nn.DataParallel(model)

    #定义优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.005)

    for i in range(EPOACH):
        optimizer.zero_grad()
        for batch_id,batch_data in enumerate(train_dataloader):
            # print(f"batch id :{batch_id}  ",end='')
            # print(f"input_ids is {batch_data[0]} and the target is {batch_data[1]}")
            input_id = batch_data[0].to(device)
            label = batch_data[1].to(device)
            #此处的input_id的size为（BATCH_SIZE/显卡数量,seq_length）
            output,loss = model(input_id,label)
            #pytorch: grad can be implicitly created only for scalar outputs
            #自动梯度下降只能是针对标量
            loss = loss.mean()
            print(f"the batch loss {loss.item()}")
            loss.backward()
            # print(f"the hidden is {output} and size is {output.size()}")
        optimizer.step()
    #此处保存的是GPU模型
    #如果要保存CPU模型，则需要定义为：torch.save(model.module,模型名称)
    torch.save(model,"./model/lstm.pt")



