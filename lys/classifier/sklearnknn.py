from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import dataprocess1.extractcharact  as extrat
from sklearn import metrics

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np

import matplotlib.pyplot as plt

# 返回一个训练完成的knn
#
from drawimg import resultOfClassifier


def getKnnClassifer(slot, k=9):
    np.set_printoptions(suppress=True)

    x, lable = extrat.normData(slot)  # 预处理数据

    # x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
    #                                                             test_size=0.6)  # 将数据集按比例划分为训练集和测试集
    knn = KNeighborsClassifier(k)
    # 定义一个knn分类器对象
    # knn.fit(x_train, lable_train.ravel())
    knn.fit(x, lable.ravel())

    return knn


# 验证分类器性能
def gradeOfClassifer(k, slot):
    np.set_printoptions(suppress=True)

    x, lable = extrat.normData(slot)  # 构造特征向量  x=[局部距离，坐标数量，均方差]  lable=[类别标签]

    x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
                                                                test_size=0.6)  # 将数据集按比例划分为训练集和测试集
    knn = KNeighborsClassifier(k)
    # 定义一个knn分类器对象
    knn.fit(x_train, lable_train.ravel())

    # score = knn.score(x_test, lable_test, sample_weight=None)
    # print(score)
    pre = knn.predict(x_test)  # 预测

    # 查准率（正确率）
    P = metrics.precision_score(lable_test, pre, average='macro')
    # 召回率
    R = metrics.recall_score(lable_test, pre, average='macro')
    # F1分数
    F1 = metrics.f1_score(lable_test, pre, average='weighted')
    # 混淆矩阵
    M = metrics.confusion_matrix(lable_test, pre, labels=[1.0, 2.0, 3.0])
    print("查准率" + str(P))
    print("召回率" + str(R))
    print("F1分数" + str(F1))
    print("混淆矩阵")
    print(M)
    print()
    return P


# 混淆矩阵
def matrixOfClassification(k, slot):
    np.set_printoptions(suppress=True)

    x, lable = extrat.normData(slot)  # 预处理数据

    # x_train:训练样本   lable_train：训练样本对应的标签
    # x_test：测试样本    lable_test ：测试样本对应的标签

    x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
                                                                test_size=0.6)  # 将数据集按比例划分为训练集和测试集
    knn = KNeighborsClassifier(k)
    # 定义一个knn分类器对象
    knn.fit(x_train, lable_train.ravel())

    score = knn.score(x_test, lable_test, sample_weight=None)
    pre = knn.predict(x_test)

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
        i = i + 1;

    print("正确率：" + str(score))
    print(lable1)
    print(lable2)
    print(lable3)
    print(lable4)
    return score


# P-R曲线
def P_R(slot, k):
    np.set_printoptions(suppress=True)

    x, lable = extrat.normData(slot)  # 预处理数据

    # x_train, x_test, lable_train, lable_test = train_test_split(x, lable, random_state=1, train_size=0.4,
    #                                                             test_size=0.6)  # 将数据集按比例划分为训练集和测试集
    knn = KNeighborsClassifier(k)

    # n_samples, n_features = x.shape
    #
    # random_state = np.random.RandomState(0)
    #
    # x = np.c_[x, random_state.randn(n_samples, 200 * n_features)]
    kfold = StratifiedKFold(n_splits=5)
    cv = kfold.split(x, lable)
    for i, (train, test) in enumerate(cv):

         knn.fit(x[train], lable[train])
         probas_ =knn.predict_proba(x[test])
         print(probas_)
         precision, recall, thresholds = precision_recall_curve(lable[test], probas_[:, 1])
         plt.plot(recall, precision, lw=1)
         break
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel("Recall Rate")

    plt.ylabel("Precision Rate")
    plt.show()


# if __name__ == '__main__':
#     seqK = [9, 20]
#
#     for k in seqK:
#         print("k=" + str(k) + "  ")
#         result = []
#         for t in range(5, 60, 5):
#             print("t=" + str(t), end="  ")
#             acc = gradeOfClassifer(k, t)
#             result.append(acc)
#         print()
#         resultOfClassifier.draw(1, "k=" + str(k), range(5, 60, 5), result)
