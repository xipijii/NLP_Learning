# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""
""" ✍️

__init__（配置，模型，日志）
1. 定义：模型、日志
2. 加载验证数据
3. 定义统计测试结果的变量

eval
1. 日志记录（第几轮的评估），开启评估模式，清空上一轮的统计结果
2. 循环验证数据开启验证和统计
    如果GPU可用，将数据复制到GPU
    通过模型得到预测值
    调用write_stats比较测试值和真实值
3. 计算、返回准确率

write_stats（测试值，真实值）
1. 断言：测试值和真实值数据量相等
2. 获取测试值中概率最大的角标，与真实值比较是否相等，统计正误

show_stats
1. 根据统计值输出正确率

"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  # 用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():  # ‼️禁用梯度计算
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
