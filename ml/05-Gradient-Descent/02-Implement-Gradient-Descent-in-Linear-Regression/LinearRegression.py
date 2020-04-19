#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 22:43
# @Author  : duanhaobin
# @File    : LinearRegression.py
# @Software: PyCharm
# @Desc    : 实现多元线性回归算法

import numpy as np


class LinearRegression:

    def __init__(self):
        """初始化分类器"""
        self.coefficient_ = None  # 系数
        self.interception_ = None  # 截距
        self.__theta = None  # 根据公式直接就能计算出

    def fit_normal(self, X_train, y_train):
        """
            根据训练数据计算出正规方程解,训练LinearRegression模型
        :X_train 样本训练数据
        :y_train 标记训练数据
        :return self
        """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 给X_train的第一列增加1列数据,且每一项都为1
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        # 根据公式计算theta
        self.__theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        # 截距和系数都分别赋值
        self.coefficient_ = self.__theta[1:]
        self.interception_ = self.__theta[0]
        return self

    def fit_gd(self,X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            res[0] = np.sum(X_b.dot(theta) - y)
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """
            给定待预测数据集X_predict,返回表示X_predict的结果向量
        :X_train 样本待预测数据
        :return 预测的结果向量
        """
        assert self.interception_ is not None and self.coefficient_ is not None, \
            "must fit before predict!"

        assert X_predict.shape[1] == len(self.coefficient_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        return X_b.dot(self.__theta)

    def score(self, X_test, y_test):
        """
            根据测试数据集测试模型得分
        :X_test 测试数据集
        :y_test 测试标记集
        :return float 得分
        """
        y_predict = self.predict(X_test)

        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
