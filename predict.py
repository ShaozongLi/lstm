# -*- coding:utf-8 -*-
# @Time: 2021/8/11 13:48
# @Author: ShaozongLi
# @Email: lishaozong@hikvision.com.cn
# @File: predict.py
import torch

from lstm.tokenizer import LstmTokenzier

if __name__=="__main__":
    text="蛋糕真好吃啊"
    tokenizer =LstmTokenzier()
    input_ids = tokenizer.token(text=text,max_length=64)
    input_ids = torch.tensor(input_ids,dtype=torch.long).view(1,-1)
    #GPU下训练的模型，在CPU下运行需要注意：
    #模型加载方式：torch.load(模型文件,map_location="cpu").module
    model = torch.load("./model/lstm.pt",map_location="cpu").module
    with torch.no_grad():
        output,loss = model(input_ids)
    print(f"the output is {output}")
    index = torch.argmax(output,dim=1)
    label = tokenizer.get_label_name(index)
    print(label)


