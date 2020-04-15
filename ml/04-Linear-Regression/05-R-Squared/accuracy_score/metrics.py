#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14 19:45
# @Author  : duanhaobin
# @File    : metrics.py
# @Software: PyCharm
# @Desc    : MSE,RMSE,MAE,R2算法封装

import numpy as np
from math import sqrt


def mean_squared_error(y_true, y_predict):
    """根据真值和预测值计算均方误差"""
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squraed_error(y_true, y_predict):
    """根据真值和预测值计算均方根误差"""
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """根据真值和预测值计算平均绝对误差"""
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """根据真值和预测值计算R2"""
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)