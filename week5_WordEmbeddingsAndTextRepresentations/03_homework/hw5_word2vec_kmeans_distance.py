#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))  # ✍️已分词
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，分成句子
        vector = np.zeros(model.vector_size)  # 128纬是哪里指定的？
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]  # ✍️对句内的所有纬度进行加和（对位相加‼️）
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))  # 纵向求平均
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化
    # ✍️根据词向量模型找到向量后，加和求平均，以获得句向量

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    # euc_distance_label_dict = defaultdict(float)
    # cos_distance_label_dict = defaultdict(float)
    euc_distance_label_dict = defaultdict(list)
    cos_distance_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  # 取出句子和标签
        # kmeans.cluester_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        # todo 计算类内平均距离（句向量与质心的距离）并记录label
        center = kmeans.cluster_centers_[label]
        euc_distance = (math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, center))))
        cos_distance = (sum(a * b for a, b in zip(vector, center)) / (
                math.sqrt(sum(a ** 2 for a in vector)) * math.sqrt(sum(b ** 2 for b in center))))
        euc_distance_label_dict[label].append(euc_distance)
        cos_distance_label_dict[label].append(cos_distance)
    for label, distance_list in euc_distance_label_dict.items():  # ✍️
        euc_distance_label_dict[label] = np.mean(distance_list)
    for label, distance_list in cos_distance_label_dict.items():  # ✍️
        cos_distance_label_dict[label] = np.mean(distance_list)

    # todo 使用欧式距离排序和打印
    print("================欧式距离================")
    labels_sentences = {}  # # 综合label，句子，和欧式距离
    for label, distance in euc_distance_label_dict.items():
        sentences = sentence_label_dict[label]
        labels_sentences[(label, distance)] = sentences
    sorted_dict = dict(sorted(labels_sentences.items(), key=lambda x: x[0][1], reverse=False))
    print([int(label) for label, distance in sorted_dict.keys()])
    for (label, distance), sentences in sorted_dict.items():
        print(f"cluster：{label}，类内平均距离：{distance}")
        for i in range(min(10, len(sentences))):  # 打印前十
            print(sentences[i].replace(" ", ""))
        print("---------")

    # todo 使用余弦距离排序和打印
    print("================余弦距离================")
    labels_sentences = {}  # # 综合label，句子，和欧式距离
    for label, distance in cos_distance_label_dict.items():
        sentences = sentence_label_dict[label]
        labels_sentences[(label, distance)] = sentences
    sorted_dict = dict(sorted(labels_sentences.items(), key=lambda x: x[0][1], reverse=True))
    print([int(label) for label, distance in sorted_dict.keys()])
    for (label, distance), sentences in sorted_dict.items():
        print(f"cluster：{label}，类内平均距离：{distance}")
        for i in range(min(10, len(sentences))):  # 打印前十
            print(sentences[i].replace(" ", ""))
        print("---------")


    print("结束")

if __name__ == "__main__":
    main()
