import logging
import os
import numpy as np
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluator import Evaluator
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

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


def get_lora_model(model):
    # 大模型微调策略
    peft_config = LoraConfig(
        r=16,  # ✍️设置越大越接近全量微调的效果，决定训练的矩阵大小
        lora_alpha=32,  # ✍️放缩因子 = lora_alpha / r
        lora_dropout=0.1,
        # target_modules=["query", "key", "value"
        #                 , "dense", "intermediate.dense"]  # ✍️在哪些地方进行放缩，根据Bert的名字来的
        target_modules=["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
        # target_modules=["query", "key", "value"]
    )
    model = get_peft_model(model, peft_config)  # ✍️配置过lora之后的model
    # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度‼️
    # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置‼️
    # ✍️这里会冻结模型整体的权重不能训练（requires_grad = True）
    # ✍️但比如一个分类任务，最后会使用一个线性层进行分类，这部分是需要训练的
    for param in model.get_submodule("model").get_submodule("classification").parameters():
        param.requires_grad = True

    return model


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    # todo 启动大模型微调策略
    model = get_lora_model(model)

    # 打印可训练参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"可训练: {name}, shape={param.shape}")

    cuda_flag = torch.cuda.is_available()
    optimizer = choose_optimizer(config, model)  # todo 更改位置
    evaluator = Evaluator(config, model, logger)
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
            optimizer.zero_grad()  # todo 更改位置
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

            train_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data


if __name__ == '__main__':
    model, train_data = main(Config)
