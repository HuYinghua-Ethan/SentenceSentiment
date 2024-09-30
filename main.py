# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import csv
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 是否用GPU进行训练
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("Cuda is available, use GPU to train.")
        model = model.cuda()
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []  # 记录训练的损失值
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            # print(batch_data)
            # input()
            optimizer.zero_grad()
            input_ids, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            # print(input_ids.shape)  # torch.Size([128, 100, 200])
            # print(labels.shape)     # torch.Size([128, 1])
            # input()
            loss = model(input_ids, labels)  # 计算损失
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item()) # loss.item() 是 PyTorch 中用于从损失张量（通常是一个标量张量）中提取标量值的方法。
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # if acc > 0.997:
        #     model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        #     torch.save(model.state_dict(), model_path)
    return acc





if __name__ == "__main__":
    main(Config)


