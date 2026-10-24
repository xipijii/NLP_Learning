import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):

    def __init__(self, input_size, output_size=1):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        linear_output = self.layer(x)
        y_pre = self.activation(linear_output)
        if y is not None:
            # return self.loss(x, y_pre)  # ‼️计算loss是y与y_pre
            return self.loss(y_pre, y)  # ‼️计算loss是y与y_pre
        else:
            return y_pre


# 构造一个数据
def generate_data():
    x = np.random.random(5)
    # y = 0
    if x[0] > x[4]:
        # y = 1
        return x, 1
    else:
        # y = 0
        return x, 0
    # return torch.FloatTensor(x), torch.FloatTensor([y])  # ❌不是在构造成1个数据时转成张量，而是一组数据


# 构造一组数据
def build_dataset(total_train_num):
    x_dataset = []
    y_dataset = []
    for i in range(total_train_num):
        # x_dataset.append(generate_data()[0])
        # y_dataset.append(generate_data()[1]) ❌y应该加上[]❓
        x, y = generate_data()
        x_dataset.append(x)
        y_dataset.append([y])
        # return x_dataset, y_dataset  ❌需要转换成张量
    return torch.FloatTensor(x_dataset), torch.FloatTensor(y_dataset)


def evaluate(model):
    model.eval()  # ⚠️📒模型开启评估模式
    test_sample_num = 100  # ⚠️
    # 1.构造一定数量的数据，获取x和y；循环
    x_dataset, y_dataset = build_dataset(test_sample_num)  # 📒这是张量
    # print(f"本次预测集中共有{sum(y_dataset)}个正样本，{test_sample_num - sum(y_dataset)}个负样本")
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y_dataset), test_sample_num - sum(y_dataset)))
    correct, wrong = 0, 0
    with torch.no_grad():  # ⚠️📒 即使没有调用计算梯度的方法，也会记录一些中间值，加了no_grad()就不会记录了，减负
        # for x, y in zip(x_dataset, y_dataset):  # ❌不是一个x, y 一个x, y单独预测的
        y_pre_dataset = model(x_dataset)  # 📒
        for y, y_pre in zip(y_dataset, y_pre_dataset):
            # 2.使用模型得到y_pre
            # y_pre = model(x)  # 📒模型输入就是张量
            # 3.对比y和y_pre
            #   如果y_pre>0.5，视为正样本1，否则视为负样本0
            #   与预测一致，记录正确，否则错误
            if y_pre > 0.5 and y == 1:
                correct += 1
            elif y_pre <= 0.5 and y == 0:
                correct += 1
            else:
                wrong += 1

    acc = correct / test_sample_num
    print(f"共评估样本量{test_sample_num}，其中正确预测{correct}个，错误预测{wrong}个，正确率{acc}")


    return acc


def main():
    # 定义参数
    epoch_num = 20
    total_train_size = 5000
    batch_size = 20
    learning_rate = 0.01
    input_size = 5
    log = []

    # 构建模型
    model = TorchModel(input_size)

    # 优化器 ‼️ ⚠️-抄的
    optim = torch.optim.Adam(model.parameters(), learning_rate)
    # 创建训练集（正常任务是读取训练集）（每一轮训练的数据都是一样的‼️❓）
    x_dataset, y_dataset = build_dataset(total_train_size)

    # 开始训练
    for i in range(epoch_num):
        model.train()
        watch_loss = []
        # count = 0
        # 输入和输出 ❓每一轮训练的数据都是一样的‼️
        # x_dataset, y_dataset = build_dataset(total_train_size)
        for index in range(total_train_size // batch_size):  # 20个数据为一组  ⚠️抄的
            # count += 1
            x = x_dataset[index * batch_size: (index + 1) * batch_size]  # ⚠️抄的
            y = y_dataset[index * batch_size: (index + 1) * batch_size]  # ⚠️抄的
            # # 输出预测值
            # y_pre = model(x)
            # ❌无需计算预测值，模型的forward函数里面已经有了（激活层的输出），只需传入真实值y即可
            # 计算loss
            # loss = model(x, y_pre) ❌
            loss = model(x, y)  # ‼️内部计算预测值，再进行loss计算
            # if count == batch_size:  # 不用20为一组计算梯度和更新权重了❓SGD-每次使用一个样本计算梯度❓
            loss.backward()  # 计算梯度 ‼️
            # 📒loss.backward()本身不关心batch_size，是loss函数，有个参数reduction默认值是mean
            optim.step()  # 更新权重 ‼️
            optim.zero_grad()  # 梯度归0 ‼️
            # watch_loss.append(loss)  # 累加loss值 ❌
            watch_loss.append(loss.item())  # 累加loss值‼️
        # print(f"第 {epoch_num + 1} 轮更新权重，loss平均值为：{watch_loss / (total_train_size // batch_size)}")  # ❌
        print(f"第 {i + 1} 轮更新权重，loss平均值为：{np.mean(watch_loss)}")  # 📒np.mean()计算平均值
        # 评估模型准确率
        acc = evaluate(model)  # 📒测试本轮的模型效果
        # 记录数据（画图原数据）
        log.append([acc, float(np.mean(watch_loss))])  # ⚠️ 抄的

    # 保存模型 ⚠️ torch里有保存模型权重的方法
    torch.save(model.state_dict(), "model.bin")
    # 画图 ⚠️ 抄的 ⚠️
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

"""
使用模型进行预测 
1.输入数据（内部生成）
2.使用模型预测类别和概率值【见控制台】
"""
def predict(model_path):
    # ⚠️ 抄的
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # ⚠️加载训练好的权重 ‼️
    model.eval()  # 测试模式
    with torch.no_grad():

        # ⚠️ 自己写的
        x_set, y_set = build_dataset(10)
        y_pre_set = model(x_set)
        for x, y_pre in zip(x_set, y_pre_set):
            # if y_pre > 0.5:
            #     label = 1
            # else:
            #     label = 0
            # label = int(y_pre > 0.5)  # 📒
            label = 1 if y_pre > 0.5 else 0  # 📒
            print(f"输入：{x.numpy()}，预测类别：{label}，概率值：{y_pre}")


if __name__ == '__main__':
    main()
    # predict("model.bin")  # ⚠️