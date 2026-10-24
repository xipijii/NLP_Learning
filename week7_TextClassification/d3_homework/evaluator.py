import torch

from loader import load_data

class Evaluator:

    def __init__(self, model, logger, config):
        self.model = model
        self.logger = logger
        self.config = config
        self.valid_data = load_data(config["valid_data_path"], config)
        self.stat_dict = {"correct": 0, "wrong": 0}

    def eval(self, epoch):
        self.logger.info(f"epoch {epoch} eval")
        self.stat_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            pre_data = self.model(input_ids)
            self.write_stat(pre_data, labels)
        acc = self.show_stat()
        return acc

    def write_stat(self, pre_data, labels):
        for y_pre, label in zip(pre_data, labels):
            max_idx = torch.argmax(y_pre)
            if int(max_idx) == int(label):
                self.stat_dict["correct"] += 1
            else:
                self.stat_dict["wrong"] += 1

    def show_stat(self):
        correct, wrong = self.stat_dict["correct"], self.stat_dict["wrong"]
        self.logger.info(f"总样本量：{correct + wrong}，正确样本总量：{correct}，错误样本总量：{wrong}")
        return correct/(correct + wrong) * 100

