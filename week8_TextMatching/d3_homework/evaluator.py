import torch
from loader import load_data
"""
模型效果测试
"""

class Evaluator:
    """
    __init__
    引入：配置、模型、日志
    加载数据：验证数据集、训练集（需要知识库）
    定义字典：记录正确和错误
    """
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}

    """
    eval
    准备工作：打日志、开启评估模式
    知识库加载：知识库转成向量（
    模型评估：根据模型encoder出向量
    比较结果：
        将模型评估结果（用户问句向量）和验证集的label（相似问句向量）计算余弦值
        取出距离中最大的极为最相似的（1-cos代表距离）
        如果最小距离的idx和label相等，则为正确
    """
    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch+1}轮模型效果：")
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return



    """
    knwb_to_vector
    定义：
        定义所有相似问对应标准问的idx（通过记录id的自然序号）
        记录问题编号到标准问标号的映射
        将id序号堆叠成更高纬度的张量，将张量迁移到gpu
        通过模型将堆叠后的张量转成向量
        将向量归一化（方便后续计算模长）
    """
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            # 命中问题编号并转化为标准问的编号
            hit_index = int(torch.argmax(res.squeeze()))
            hit_index = self.question_index_to_standard_question_index[hit_index]
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return

