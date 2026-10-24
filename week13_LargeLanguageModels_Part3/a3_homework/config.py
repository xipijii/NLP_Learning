# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "pretrain_model_path": r"/Users/juju/PycharmProjects/py_project01/nlp_project02/week6_LanguageModelsAndPretraining/01_reference/bert-base-chinese",
    "bert_vocab_path": r"/Users/juju/PycharmProjects/py_project01/nlp_project02/week6_LanguageModelsAndPretraining/01_reference/bert-base-chinese/vocab.txt",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "tuning_tactics": "lora_tuning",  # todo
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": True,
    "use_bert": True,
    "class_num": 9,
    "bert_path": r"/Users/juju/PycharmProjects/py_project01/nlp_project02/week6_LanguageModelsAndPretraining/01_reference/bert-base-chinese"
}

