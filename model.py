# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["embedding_dim"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["embedding_dim"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

        if model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        

        self.linear = nn.Linear(hidden_size, class_num)
        if config["vocab_type"] == "words":
            self.pooling_style = "avg"
        elif config["vocab_type"] == "chars":
            self.pooling_style = config["pooling_style"]

        self.linear = nn.Linear(hidden_size, class_num)
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失
    
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x, target=None):
        if isinstance(x, tuple):  #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        # 可以采用pooling的方式得到句向量
        if self.pooling_style == "avg":
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        elif self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)

        predict = self.linear(x)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)



