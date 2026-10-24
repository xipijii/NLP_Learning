import logging
import os.path
import pandas as pd
import random
import torch
import numpy as np

from config import Config
from loader import load_data, split_dataset
from model import TorchModel, choose_optimizer
from evaluator import Evaluator

"""
构建日志，设置种子
1. 创建保存模型的目录
2. 加载训练数据
3. 定义模型、优化器、评估方法
4. 是否使用GPU，是则迁移模型到GPU上
5. 开启训练
（1）获取和迁移批次数据‼️
（2）模型训练获取loss
（3）反向传播、梯度更新、梯度归0 ‼️顺序别搞混，先反向传播再梯度更新
（4）记录loss，打印（%，当前轮）
（5）验证准确率并返回
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    epoch_num = config["epoch_num"]

    split_dataset(config["dataset_path"], config)
    train_dataset = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    optimizer = choose_optimizer(model, config)
    evaluator = Evaluator(model, logger, config)

    use_flag = torch.cuda.is_available()
    if use_flag:
        model.cuda()

    model.train()
    for epoch in range(epoch_num):
        logger.info(f"epoch {epoch + 1} begin")
        watch_loss = []
        for index, batch_data in enumerate(train_dataset):
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()  # ❓做了啥
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

            if index % (len(train_dataset) // 5) == 0:
                logger.info(f"进度 {index / (len(train_dataset)) * 100:.2f} %，当前batch_loss = {loss.item()}")

        logger.info(f"本轮的loss平均值={np.mean(watch_loss)}")
        # acc = evaluator(epoch)  # ❌
        acc = evaluator.eval(epoch)
        logger.info(f"本轮的准确率={acc:.2f}")

        return acc

# AI生成
def save_results_to_excel(results_list, output_dir="output"):
    """将实验结果保存到 Excel 文件"""
    # 确保 output 文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建 DataFrame
    df = pd.DataFrame(results_list)

    # 生成文件名（带时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.xlsx"
    output_path = os.path.join(output_dir, filename)

    # 保存到 Excel
    df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"\n✅ 实验结果已保存到: {output_path}")
    print(f"   共 {len(results_list)} 条实验记录")

    return output_path


if __name__ == '__main__':
    results_list = []
    # ✍️fast_text, lstm, gru?, rnn, cnn, gated_cnn, stack_gated_cnn, rcnn
    # bert, bert_lstm, bert_cnn, bert_mid_layer
    # for model in ["bert"]:  # todo
    for model in [
        "lstm", "rnn", "gated_cnn",
        "bert"
        , "bert_lstm", "bert_mid_layer"
    ]:  # todo
        Config["model_type"] = model
        # for lr in [1e-5]:
        for lr in [1e-3, 1e-4, 1e-5]:
            Config["learning_rate"] = lr
            # for batch_size in [20]:
            for batch_size in [20, 50, 100]:
                Config["batch_size"] = batch_size
                for hidden_size in [128, 256, 512]:
                    Config["hidden_size"] = hidden_size
                    acc = main(Config)
                    print(f"最后一轮准确率：{acc:.2f}%，当前配置：{Config}")
                    results_list.append({
                        "model_type": model,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "hidden_size": hidden_size,
                        "accuracy": acc
                    })
    save_results_to_excel(results_list)

        # todo
        # 打印【句子，真实标签，测试结果，是否一致，测试分布】
        # todo
        # 打印【参数，准确率】
