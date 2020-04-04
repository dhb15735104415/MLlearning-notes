#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/1/17 10:25
# @Author  : duanhaobin
# @File    : kNN.py
# @Software: PyCharm
# @Desc    :   简单封装自己实现的kNN算法

import numpy as np
from math import sqrt
from collections import Counter


def kNNclassify(k, X_train, y_train, x):
    """
    封装自己实现的kNN算法
    @param k: k值
    @param X_train: 样本训练集
    @param y_train: 特征训练集
    @param x: 需要预测的数据
    @return: 返回预测结果
    """
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
    nearest = np.argsort(distances)
    topK_y = [y_train[near] for near in nearest[:k]]

    votes = Counter(topK_y)
    return votes.most_common(1)[0][0]
