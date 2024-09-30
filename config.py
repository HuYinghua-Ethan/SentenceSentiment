# -*- coding: utf-8 -*-


"""
配置参数信息
"""


Config = {
    "model_path": "model",
    "train_data_path": r"./data/train.txt",
    "test_data_path": r"./data/test.txt",
    "char_vocab_path": r"./data/chars.txt",
    "words_vocab_path": r"./data/tencent_embedding_word2vec.txt",
    "vocab_type": "words", # words or chars
    "model_type":"cnn",
    "max_seq_length": 100,
    "embedding_dim": 200,
    # "hidden_size": 256,
    "kernel_size": 3, 
    "num_layers": 2,
    "epochs": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987
}









