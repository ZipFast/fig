import numpy as np
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

import dataprocess1.extractcharact  as extrat

import datetime
import os

# 返回svm分类器对象
from drawimg import resultOfClassifier


def getSvmClassifer(slot):
    np.set_printoptions(suppress=True)
    x, lable = extrat.normData(slot)

    # x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
    #                                                             test_size=0.6)  # 将数据集按比例划分为训练集和测试集

    clf = svm.SVC(C=2, kernel="linear", gamma=20, decision_function_shape='ovr')

    # clf.fit(x_train, lable_train.ravel())  # 提供训练集和标签 训练svm

    clf.fit(x, lable.ravel())  # 提供训练集和标签 训练svm
    return clf


# svm准确率

def gradeOfClassifer(kenel, slot):
    np.set_printoptions(suppress=True)
    x, lable = extrat.normData(slot)

    x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
                                                                test_size=0.6)  # 将数据集按比例划分为训练集和测试集
    # if kenel == 0:
    #     clf = svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape='ovr')
    # elif kenel == 1:
    #   clf = svm.SVC(C=0.8, kernel="linear", gamma=20, decision_function_shape='ovr')
    # elif kenel == 2:
    #     clf = svm.SVC(C=0.8, kernel="poly", gamma=20, degree=3, decision_function_shape='ovr')
    # else:
    #     clf = svm.SVC(C=0.8, kernel="sigmoid", gamma=20, decision_function_shape='ovr')
    # clf = svm.SVC(C=0.8, kernel="poly", gamma=20, degree=3, decision_function_shape='ovr')
    clf = svm.SVC(C=0.8, kernel="linear", gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, lable_train.ravel())  # 提供训练集和标签 训练svm
    pre = clf.predict(x_test)  # 预测

    # 查准率（正确率）
    P = metrics.precision_score(lable_test, pre, average='macro')
    # 召回率
    R = metrics.recall_score(lable_test, pre, average='macro')
    # F1分数
    F1 = metrics.f1_score(lable_test, pre, average='weighted')
    # 混淆矩阵
    M = metrics.confusion_matrix(lable_test, pre, labels=[1.0, 2.0, 3.0, 4.0])
    print("查准率" + str(P))
    print("召回率" + str(R))
    print("F1分数" + str(F1))
    print("混淆矩阵")
    print(M)
    print()
    return P


def matrixOfClassification(kenel, slot):
    np.set_printoptions(suppress=True)
    x, lable = extrat.normData(slot)

    x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
                                                                test_size=0.6)  # 将数据集按比例划分为训练集和测试集

    if kenel == 0:
        clf = svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape='ovr')
    elif kenel == 1:
        clf = svm.SVC(C=0.8, kernel="linear", gamma=20, decision_function_shape='ovr')
    # elif kenel == 2:
    #     clf = svm.SVC(C=0.8, kernel="poly", gamma=20, degree=2, decision_function_shape='ovr')
    else:
        clf = svm.SVC(C=0.8, kernel="sigmoid", gamma=20, decision_function_shape='ovr')

    clf.fit(x_train, lable_train.ravel())  # 提供训练集和标签 训练svm
    score = clf.score(x_test, lable_test)
    pre = clf.predict(x_test)

    # 四个数组代表四个类别的分类情况
    # lable1[1]表示类1的正确分类数，lable1[2]表示类1的被判定为类2，lable1[3]表示类1的被判定为类3，lable1[4]表示类1的被判定为类4
    # 其他三个数组同上
    lable1 = [0, 0, 0, 0, 0]
    lable2 = [0, 0, 0, 0, 0]
    lable3 = [0, 0, 0, 0, 0]
    lable4 = [0, 0, 0, 0, 0]

    i = 0
    while i < len(pre):
        label = int(pre[i])  # 类型：1.0   2.0   3.0   4.0  转整形
        if lable_test[i] == 1.0:
            lable1[label] += 1

        elif lable_test[i] == 2.0:
            lable2[label] += 1

        elif lable_test[i] == 3.0:
            lable3[label] += 1

        elif lable_test[i] == 4.0:
            lable4[label] += 1

        i = i + 1

    print("正确率：" + str(score))
    print(lable1)
    print(lable2)
    print(lable3)
    print(lable4)
    return score

if __name__ == '__main__':
    result = []
    print("kernel=" + "linear")
    for t in range(5, 60, 5):
        print("t=" + str(t), end="  ")
        acc = gradeOfClassifer(1, t)
        result.append(acc)
    print()
    resultOfClassifier.draw(2, "kernel=" + "linear", range(5, 60, 5), result)
