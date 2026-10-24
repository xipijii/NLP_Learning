import logging
import os
import numpy as np
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluator import Evaluator

import torch.cuda
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
1. 创建保存模型的目录
2. 加载：训练数据、模型、优化器、验证器
3. 模型迁移至GPU
4. 训练
5. 保存模型
"""


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型到gpu")
        model = model.cuda()
    """
    训练过程
    1. 启动：开启模型训练模式、记录log、定义loss记录
    2. 循环训练数据：
        迁移数据到gpu
        获取输入id和label
        通过模型计算loss
        反向传播、梯度更新、梯度归0
    3. 记录本轮loss和打印loss均值
    4. 验证本轮模型效果
    """
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data


if __name__ == '__main__':
    model, train_data = main(Config)
