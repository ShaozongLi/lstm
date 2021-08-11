# -*- coding:utf-8 -*-
# @Time: 2021/8/10 16:22
# @Author: ShaozongLi
# @Email: lishaozong@hikvision.com.cn
# @File: tokenizer.py

class LstmTokenzier():
    def __init__(self,):
        with open("./data/vocab.txt",'r',encoding='utf-8') as f:
            lines = f.readlines()
            self.vocab = [line.replace('\n','') for line in lines]
        with open("./data/labels.txt",'r',encoding='utf-8') as f:
            lines = f.readlines()
            self.labels = [line.replace('\n','') for line in lines]

    def get_vocab(self):
        data_url = "./data/train.tsv"
        vocab = set()
        labels = set()
        with open(data_url,'r',encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line_list = list(line.split('\t')[0])
            for word in line_list:
                if word not in vocab:
                    vocab.add(word)
            label = line.replace('\n','').split('\t')[1]
            if label not in labels:
                labels.add(label)
        with open('./data/vocab.txt','w',encoding='utf-8') as f:
            for word in vocab:
                f.writelines(word+"\n")

        with open('./data/labels.txt','w',encoding='utf-8') as f:
            for label in labels:
                f.writelines(label+"\n")


    def token(self,text,max_length=128):
        word_list = list(text)
        input_id = [self.vocab.index(word)+1 for word in word_list]
        if len(input_id)<max_length:
            input_id.extend([0] * (max_length - len(input_id)))
        else:
            input_id = input_id[:max_length]
        return input_id

    def get_label(self,label):
        return self.labels.index(label)

    def get_label_name(self,index):
        return self.labels[index]

if __name__=="__main__":
    tokenizer = LstmTokenzier()
    tokenizer.get_vocab()
    input_id = tokenizer.token("你叫什么名字",32)
    print(input_id)

