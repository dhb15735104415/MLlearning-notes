#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/2/10 12:16
# @Author  : duanhaobin
# @File    : model_selection.py
# @Software: PyCharm
# @Desc    : 分割训练数据和测试数据模块

import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """
    将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test
    """
    assert X.shape[0] == y.shape[0], \
        "the size of X must be eaqul to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"
    if seed:
       np.random.seed(seed)

    # 随机置换序列,这样就能保证特征值和标记一一对应
    shuffled_indexes = np.random.permutation(len(X))

    # 通过分割比例ratio,设置测试和训练数据
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
