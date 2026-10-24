import logging
import os
import numpy as np
import torch.cuda

from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluator import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
1. 创建保存模型的记录
2. 加载训练数据
3. 加载模型
4. 标识是否使用gpu
5. 加载优化器
6. 加载效果测试类
7. 训练
（1）准备：开启训练模式、记录日志、定义loss集合
（2）取出batch大小的数据，进行循环：
    a. 如果gpu可用，将数据迁移到gpu
    b. 获取三个输入
    c. 使用模型计算loss
    d. 添加loss
    e. 反向传播：计算梯度、梯度更新、梯度归0
（3）记录日志：本轮的loss平均值
（4）评估本轮的模型
8. 保存模型
"""


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = SiameseNetwork(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可用，迁移模型至gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    # 开启训练
    for epoch in range(config["epoch"]):
        model.train()
        watch_loss = []
        logger.info(f"epoch {epoch + 1} begin")
        for batch_data in train_data:
            input_id1, input_id2, input_id3 = batch_data
            loss = model(input_id1, input_id2, input_id3)
            watch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"本轮的loss平均值={np.mean(watch_loss)}")
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == '__main__':
    main(Config)
