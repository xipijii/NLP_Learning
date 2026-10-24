"""
配置参数信息
"""

Config = {
    "seed": 987,
    "model_path": "output",
    "model_type": "bert",
    "pretrain_model_path": r"/Users/juju/PycharmProjects/py_project01/nlp_project02/week6_LanguageModelsAndPretraining/01_reference/bert-base-chinese",
    "dataset_path": "文本分类练习.csv",
    "train_data_path": "train_data.txt",
    "valid_data_path": "valid_data.txt",
    "vocab_path": "chars.txt",

    "vocab_size": 10000,
    "max_length": 30,
    "optimizer": "adam",

    "epoch_num": 20,
    "batch_size": 50,
    "train_sample_num": 10000,

    "layer_num": 12,
    "learning_rate": 1e-3,
    "hidden_size": 128,
    "max_len": 512,
    "class_num": 2,
    "pooling_type": "avg",
    "kernel_size": 3
}
