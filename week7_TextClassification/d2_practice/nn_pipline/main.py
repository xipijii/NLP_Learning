import logging
import random
# import config
from config import Config  # ✅
from model import TorchModel, choose_optimizer
from loader import load_data
from evaluator import Evaluator
import numpy as np
import torch
import os

"""
一 日志配置
二 设置随机种子（保证模型初始化参数、数据打乱顺序、Dropout 等随机操作相同 ✅ 最终训练结果可以复现）
三 定义主函数（训练函数）
    创建保存模型的目录
    构造训练数据、模型、优化器和评估类
    开启训练（循环epoch
        获取批次数据
        反向传播、梯度更新、梯度归零
        记录loss，打印日志
        评估模型准确率
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

"""
创建保存模型的目录
    构造训练数据、模型、优化器和评估类
    标识是否使用GPU，可以使用则迁移模型 ‼️
    开启训练（循环epoch
        获取批次数据（‼️如果可以使用GPU，则将数据转移到GPU显存中
        反向传播、梯度更新、梯度归零
        记录loss，打印日志
        评估模型准确率
"""


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # batch_size = Config["batch_size"]  # ❌Dataloader不需要
    # train_sample_num = Config["train_sample_num"]  # ❌Dataloader不需要
    # watch_loss = []  # ❌位置错了，应该在每个epoch里面
    train_dataset = load_data(config["train_data_path"], config)  # ‼️先加载数据再定义模型，为了获取分类类别数量
    model = TorchModel(config)
    optimizer = choose_optimizer(model, config)
    evaluator = Evaluator(config, model, logger)

    # 🤦‼️标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    model.train()

    for epoch in range(Config["epoch_num"]):
        logger.info(f"epoch {epoch + 1} begin, model {Config["model_type"]} ")
        watch_loss = []
        # for index, batch_data in enumerate(train_sample_num // batch_size):  # ❌Dataloader不是这么用的
        for index, batch_data in enumerate(train_dataset):
            #  🤦‼️ GPU复制数据
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
            # if index % (epoch / 2) == 0:
                #  ❌训练数据训练进度的0%/50%/100%
                # / 除法结果是消暑
                # print(f"进度 {index % (epoch / 2)}%，loss={loss}")  # ❌使用日志打印
            # if index % (len(batch_data) // 2) == 0:
                # logger.info(f"进度 {index % (epoch / 2)}%，loss={loss}")
            if index % (len(train_dataset) // 5) == 0:
                logger.info(f"进度 {index / len(train_dataset) * 100:.2f}%，loss={loss.item():.4f}")
        # print(f"第 {epoch} 轮，loss平均值={np.mean(watch_loss)}")  # ❌使用日志打印
        logger.info(f"batch loss = {np.mean(watch_loss)}")

        # acc = evaluator(model)  # ❌模型传入靠定义的时候传入，现在只需要传入第几轮次来进行打印

        acc = evaluator.eval(epoch)

    return acc


if __name__ == '__main__':
    """
    模型的类型
    epoch_num、batch_size、train_sample_num
    学习率
    hidden_size、max_len
    模型特殊参数：
        如果是Bert：
    """
    for bs in [20, 50]:
        Config["batch_size"] = bs
        for model in ["bert", "rnn", "lstm"]:  # ‼️
            Config["model_type"] = model
            for lr in [1e-3, 1e-4]:
                Config["learning_rate"] = lr
                for hs in [128, 256]:
                    Config["hidden_size"] = hs
                    # 定制化
                    if "cnn" in Config["model_type"]:
                        for ks in [3, 5, 7]:
                            Config["kernel_size"] = ks
                    if "bert" in Config["model_type"]:
                        for ln in [2, 6, 12]:
                            Config["num_layers"] = ln
                    print(f"最后一轮准确率：{main(Config)}，当前配置：{Config}")
# for en in [20, 40]:
#     Config["epoch_num"] = en
# for ml in [30, 60, 90]:
#     Config["max_length"] = ml
# for pt in ["avg", "max"]:
#     Config["pooling_type"] = pt
# for op in ["adam", "sgd"]:
#     Config["optimizer"] = op