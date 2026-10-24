import torch

from loader import load_data

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
3. 统计

write_stats（测试值，真实值）
1. 断言：测试值和真实值数据量相等
2. 获取测试值中概率最大的角标，与真实值比较是否相等，统计正误

show_stats
1. 根据统计值输出正确率
2. 返回准确率
"""

class Evaluator:

    def __init__(self, config, model, logger):
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config)
        self.stat_dicts = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info(f"====第 {epoch + 1} 轮评估====")

        self.model.eval()
        self.stat_dicts = {"correct": 0, "wrong": 0}
        with torch.no_grad():
            # ❌要清楚，这里的验证集数据也是batch的
            # for input_ids, label in self.valid_data:
            #     y_pre = self.model(input_ids)
            #     y_pre_dataset.append(y_pre)
            #     y_true_dataset.append(label)
            #
            # self.writ_stats(y_pre_dataset, y_true_dataset)
            for batch_data in self.valid_data:
                if torch.cuda.is_available():  # ✍️‼️
                    batch_data = [d.cuda() for d in batch_data]
                # for input_ids, labels in batch_data:
                input_ids, labels = batch_data
                with torch.no_grad():
                    y_pre_dataset = self.model(input_ids)
                self.writ_stats(y_pre_dataset, labels)
            acc = self.show_stats()  # ‼️注意位置，是在各batch数据训练完毕之后
        return acc

    # 只做记录
    def writ_stats(self, y_pre_dataset, y_true_dataset):
        assert len(y_pre_dataset) == len(y_true_dataset)
        correct, wrong = 0, 0
        for y_pre, label in zip(y_pre_dataset, y_true_dataset):
            max_idx = torch.argmax(y_pre)
            # if max_idx == label:  # ❌
            if int(max_idx) == int(label):
                self.stat_dicts["correct"] += 1
            else:
                self.stat_dicts["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stat_dicts["correct"]
        wrong = self.stat_dicts["wrong"]

        self.logger.info(f"======总样本量{correct + wrong}，正确数量{correct}，错误本量{wrong}")
        self.logger.info(f"===本轮正确率 {correct / (correct + wrong) * 100:.2f} %")
        self.logger.info("--------------------")
        return correct / (correct + wrong)
