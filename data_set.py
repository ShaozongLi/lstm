# -*- coding:utf-8 -*-
# @Time: 2021/8/10 16:48
# @Author: ShaozongLi
# @Email: lishaozong@hikvision.com.cn
# @File: data_set.py
import torch
from torch.utils.data.dataset import Dataset

from lstm.tokenizer import LstmTokenzier


class LstmDataSet(Dataset):
    def __init__(self,data_url):
        self.data_url = data_url
        with open(self.data_url,'r',encoding='utf-8') as f:
            self.lines = f.readlines()
        self.tokenzier = LstmTokenzier()


    def __getitem__(self, item):
        text,label = (self.lines[item].replace("\n",'').split('\t'))
        input_id = self.tokenzier.token(text,64)
        label = self.tokenzier.get_label(label)
        input_id = torch.tensor(input_id,dtype=torch.long)
        label = torch.tensor(label,dtype=torch.long)
        return input_id,label

    def __len__(self):
        return len(self.lines)

if __name__ =="__main__":
    dataset = LstmDataSet("./data/train.tsv")
    print(dataset[1])
