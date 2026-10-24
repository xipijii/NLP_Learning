# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from c4_model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
# [DEBUG, INFO, WARNING, ERROR, CRITICAL]

# ❓
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ✍️level=logging.INFO 设置日志的最低显示级为INFO；asctime-时间戳；name-日志器名称；levelname-日志级别；message-日志消息
logger = logging.getLogger(__name__)
    # ✍️创建日志器对象 __name__ 是当前模块的名称

"""
模型训练主程序
"""

# 设置种子 (main.py 第 26-30 行)
#     ↓
# ┌───────────────────────────────────────┐
# │ 1. random.seed(seed)                  │
# │    → DataLoader shuffle 顺序          │
# │                                       │
# │ 2. np.random.seed(seed)               │
# │    → NumPy 随机操作（如果有）          │
# │                                       │
# │ 3. torch.manual_seed(seed)            │
# │    → Embedding 权重初始化 ⭐           │
# │    → LSTM/CNN/Linear 权重初始化 ⭐     │
# │    → DataLoader shuffle 顺序          │
# │    → Dropout 随机丢弃（如果有）        │
# │                                       │
# │ 4. torch.cuda.manual_seed_all(seed)   │
# │    → GPU 上的所有随机操作              │
# └───────────────────────────────────────┘
#     ↓
# 创建模型 (main.py 第 39 行)
#     → TorchModel(config)
#     → 所有层权重根据种子初始化
#     ↓
# 加载数据 (main.py 第 37 行)
#     → load_data(...)
#     → DataLoader shuffle 顺序根据种子确定
#     ↓
# 训练循环 (main.py 第 50-70 行)
#     → 每个 batch 的数据顺序固定
#     → 前向传播、反向传播都基于相同的初始权重
#     ↓
# 最终结果
#     → 准确率、Loss 都可以复现 ✅
seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
    #
torch.cuda.manual_seed_all(seed)


"""
创建保存模型的目录
加载训练数据
加载模型
标识是否使用GPU
加载优化器
加载效果测试类
训练
    获取批次数据（使用GPU时，数据
    反向传播、梯度更新、梯度归零
    记录loss、打印日志（记录0%，50%和100%的loss值；记录批次的loss平均值）
    评估准确率
"""
def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)  # ✍️所有权重的初始化都受种子控制
        # ✍️先加载数据再定义模型，为了获取分类类别数量
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]  # ❓
                    # ✍️：将数据从 CPU 内存转移到 GPU 显存，以加速训练

            # ✍️梯度归零、计算loss、反向传播、梯度更新
            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:  # ✍️控制日志输出频率，避免打印太多信息
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    main(Config)  # ❓

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索 ‼️
    # for model in ["gated_cnn", 'bert', 'lstm']:
    for model in ["bert_mid_layer"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            # for hidden_size in [128]:
            #     Config["hidden_size"] = hidden_size
            #     for batch_size in [64, 128]:
            #         Config["batch_size"] = batch_size
            #         for pooling_style in ["avg", 'max']:
            #             Config["pooling_style"] = pooling_style
            print("最后一轮准确率：", main(Config), "当前配置：", Config)
