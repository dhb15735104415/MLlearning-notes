#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 14:51
# @Author  : duanhaobin
# @File    : metrics.py
# @Software: PyCharm
# @Desc    :


def accuracy_score(y_true, y_predict):
    """
    计算y_true和y_predict的准确率
    y_true:提供的结果真值集
    y_predict:模型预测的结果集
    """
    assert y_true.shape[0] == y_predict.shape[0], \
        "the size of the y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)