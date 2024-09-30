# -*- coding: utf-8 -*-

import json
import re
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from gensim.models import KeyedVectors


class MyDataset:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)  # 类别数
        # if self.config["model_type"] == "bert":
        #     self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["char_vocab_path"])
        self.config["vocab_size"] = len(self.vocab)   # 记录字表的大小
        self.load()

    def load(self):
        self.data = []
        words_vector_model = KeyedVectors.load_word2vec_format(self.config["words_vocab_path"], binary=False)
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                sentence = line[-1]
                # print(sentence)
                # input()
                label = int(line[0])
                sentence_vector = self.sentence_to_vector(sentence, words_vector_model)
                # print(sentence_vector)
                # input()
                sentence_vector = torch.FloatTensor(sentence_vector)
                label_index = torch.LongTensor([label])
                self.data.append([sentence_vector, label_index])
        return
    
    # 将句子转化为向量
    def sentence_to_vector(self, sentence, model):
        if self.config["vocab_type"] == "chars":
            input_id = []
            sentence = sentence.split(" ")
            sentence = "".join(sentence)
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
                input_id = self.padding(input_id)
            return input_id
        elif self.config["vocab_type"] == "words":
            word_list = [word for word in sentence.split(' ')]
            max_words = self.config["max_seq_length"]  # 截取100个字
            embedding_dim = 200
            embedding_matrix = np.zeros((max_words, embedding_dim))
            for index, word in enumerate(word_list):
                try:
                    embedding_matrix[index] = model[word]
                except:
                    pass
            return embedding_matrix


    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_seq_length"]]
        input_id += [0] * (self.config["max_seq_length"] - len(input_id))
        return input_id


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



# 加载字表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()  # token = line.strip() 去除每行内容的空白字符（如空格、换行符等）。
            token_dict[token] = index + 1 # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    dataset = MyDataset(data_path, config) # 第一步：构建 Dataset 对象
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)  # 第二步：通过Dataloader来构建迭代对象
    return dataloader


if __name__ == "__main__":
    from config import Config
    data_path = Config["train_data_path"]
    dataset = MyDataset(data_path, Config)
    print(dataset[0])

