#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 13:09
# @Author  : duanhaobin
# @File    : preprocessing.py
# @Software: PyCharm
# @Desc    : 自己封装数据预处理模块,包含标准值归一化

import numpy as np


class StandardScaler:

    def __init__(self):
        """初始化分类器"""
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
            根据训练数据X获取数据的均值和方差
        :X 训练数据集
        :return 各个特征的均值和方差
        """
        assert X.ndim == 2, "The dimension of X must be 2"

        self.mean_ = [np.mean(X[:, i]) for i in range(X.shape[1])]
        self.scale_ = [np.std(X[:, i]) for i in range(X.shape[1])]

    def transform(self, X):
        """
            将X根据这个StandardScaler进行均值方差归一化处理
        :X 数据集
        :return 归一化处理后的数据,返回列表
        """
        assert X.ndim == 2, "The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None, \
            "must fit before transform"
        assert X.shape[1] == len(self.mean_), "the feature number of X must be equal to mean_ and std_"
        resX = np.empty(shape=X.shape, dtype=float)  # 生成空矩阵,类型为float,形状和X一致
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
