# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试

"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.test_data = load_data(config["test_data_path"], config, shuffle=True)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.test_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                predResults = self.model(input_id)
            self.write_stats(predResults, labels)
        acc = self.show_stats()
        return acc

    def write_stats(self, predResults, labels):
        assert len(labels) == len(predResults)
        for pred_result, true_label in zip(predResults, labels):
            pred_label = torch.argmax(pred_result)
            if int(pred_label) == int(true_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量: %d" %(correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)


