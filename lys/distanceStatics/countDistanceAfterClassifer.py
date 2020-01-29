# 基于运动状态的距离统计
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
from classifier.mlpclassifier import getMplClassifer
from classifier.mysvm1 import getSvmClassifer
from classifier.sklearnknn import getKnnClassifer
from dataprocess1.extractcharact import getExtract

# 判定轨迹的各子序列的状态
# 参数1：分类器对象
# 参数2：时间间隔
# 参数3：轨迹数据路径
from drawimg.resultOfClassifier import draw3, draw1, draw4

# 采用分类器判定轨迹子序列状态
from evaluate import getMae, Mape, Rmse
from filter.examples.sine_wave import kalmanfilter
from filter.movinAverage import MoveAverage
from filter.sampleFilter import last_polyfit


def classifer(classifilter, slot, path):
    resultSeqStatic = []  # 静止状态
    resultSeqUnrealized = []  # 无意识运动状态
    resultSeqBackMove = []  # 短距离往返运动
    resultSeqforward = []  # 长距离往返运动

    # 运动状态分析，与剔除
    with open(path, 'r') as f:
        timeSequen = f.readlines()  # 读取整个文件数据
        slopSequen = []  # 保存当前正在处理的子序列
        flag = False  # 是否第一次记录起始时间
        start = ''  # 当前子序列的起始时间
        for s in timeSequen:  # 按行读取数据
            if len(s) <= 1:  # 跳过空行
                continue
            else:
                s = s[:-1]  # 去掉换行符
                seq = s.split(" ")  # 按"\t"切分字符
                if flag == False:
                    start = seq[4] + " " + seq[5]  # 本子数据序列第一个点采集时间
                    flag = True
                now = seq[4] + " " + seq[5]  # 当前点采集时间
                subt = (datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(start,
                                                                                                          "%Y-%m-%d %H:%M:%S")).seconds
                # 逐个获取序列
                if subt > slot:
                    # 子序列状态判定
                    if len(slopSequen) > 0:
                        vector = getExtract(slopSequen)  # 计算序列特征[[v1,v2,v3]]
                        state = classifilter.predict(vector)[0]  # 判断当前序列的运动状态
                        if int(state) == 1:
                            resultSeqStatic += slopSequen
                        elif int(state) == 2:
                            resultSeqUnrealized += slopSequen
                        elif int(state) == 3:
                            resultSeqBackMove += slopSequen
                        elif int(state) == 4:
                            resultSeqforward += slopSequen

                    # 后续处理
                    slopSequen = []
                    temp = []
                    temp.append(float(seq[1].split(":")[1]))
                    temp.append(float(seq[2].split(":")[1]))
                    slopSequen.append(temp)
                    start = now
                else:
                    temp = []
                    temp.append(float(seq[1].split(":")[1]))
                    temp.append(float(seq[2].split(":")[1]))
                    slopSequen.append(temp)

    # 最后一个子序列的状态（其长度可能比slot小）
    # if len(slopSequen) > 0:
    #     vector = getExtract(slopSequen)  # 计算序列特征[[v1,v2,v3]]
    #     state = classifilter.predict(vector)[0]  # 判断当前序列的运动状态
    #     if int(state) == 1:
    #         resultSeqStatic += slopSequen
    #     elif int(state) == 2:
    #         resultSeqUnrealized += slopSequen
    #     elif int(state) == 3:
    #         resultSeqBackMove += slopSequen
    #     elif int(state) == 4:
    #         resultSeqforward += slopSequen
    return resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward


# 计算一个序列对应的距离
# 参数：序列
def countdistance(seq):
    distance = 0.0
    # 计算一个序列的运动距离
    if len(seq) <= 1:  # 只有一个点时，距离为零
        return distance
    else:
        p1 = seq[0]  # 轨迹的第一个点
        for value in seq[1:]:
            p2 = value
            d = np.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)  # 两点间的距离
            if d < 2.0:
                distance += d
            p1 = p2
        distance = float("%.2f" % distance)
        return distance


# 卡尔曼滤波器
def kalmanFilter(seq):
    x_value = []
    y_value = []
    for i in range(len(seq)):
        temp = seq[i]
        x_value.append(float(temp[0]))
        y_value.append(float(temp[1]))
    x_value = kalmanfilter(x_value, len(x_value))  # 滤波

    length = min(len(x_value), len(y_value))  # 数组长度
    sequence = [[x_value[i], y_value[i]] for i in range(0, length)]
    return sequence


# 最小二乘法拟合多项式
def leastfit(seq):
    result = last_polyfit(seq)
    return result


def MovingAverage(seq):
    obj = MoveAverage(seq, 11)  # 创建对象
    result = obj.smooth()  # 平滑数据

    return result


# 误差曲线图
def main(slot, other=15):
    dirPath = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\data"
    path = [dirPath + '\\' + x for x in os.listdir(dirPath)]

    # 分类其模型
    # classfilter = getKnnClassifer(slot, other)

    classfilter = getSvmClassifer(slot)

    #
    # classfilter = getMplClassifer(slot)
    result = []
    result1 = []
    result2 = []
    for p in path:
        distance = 0.0  # 去除静止状态
        distance1 = 0.0  # 除去静止状态和无意识状态后的轨迹的距离
        distance2 = 0.0  # 除去静止状态后，在拟合处理数据的轨迹的距离
        # 各状态下的轨迹
        resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward = classifer(classfilter, slot, p)

        # 除去静止状态后的轨迹的距离
        distance = countdistance(resultSeqUnrealized) + countdistance(resultSeqBackMove) + countdistance(
            resultSeqforward)

        result.append(distance)

        # 除去静止状态和无意识状态后的轨迹的距离
        distance1 = countdistance(resultSeqBackMove) + countdistance(resultSeqforward)
        result1.append(distance1)

        # #除去静止状态后，在拟合处理数据的轨迹的距离
        distance2 = countdistance(MovingAverage(resultSeqUnrealized)) + countdistance(
            MovingAverage(resultSeqBackMove)) + countdistance(
            MovingAverage(resultSeqforward))
        result2.append(distance2)

    return result, result1, result2


# index=1  卡尔曼滤波
# index=2  最小二乘
# index=1  滑动均值
def deal(index, seq):
    if index == 1:
        return kalmanFilter(seq)
    elif index == 2:
        return leastfit(seq)
    elif index == 3:
        return MovingAverage(seq)


# value:真实值
# 使用三种方法分别处理除静止之外的数据
def main1(value, slot, other=15):
    dirPath = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\data"
    path = [dirPath + '\\' + x for x in os.listdir(dirPath)]

    # 分类其模型
    # classfilter = getKnnClassifer(slot, other)

    # classfilter = getSvmClassifer(slot)

    #
    classfilter = getMplClassifer(slot)

    print("无意识运动状态  " + "往返  " + "前行")
    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            for k in [1, 2, 3]:
                distance2 = 0.0  #
                for p in path:
                    # 各状态下的轨迹
                    resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward = classifer(classfilter,
                                                                                                          slot, p)
                    # #除去静止状态后，在拟合处理数据的轨迹的距离
                    d1 = countdistance(deal(i, resultSeqUnrealized))
                    d2 = countdistance(deal(j, resultSeqBackMove))
                    d3 = countdistance(deal(k, resultSeqforward))
                    distance2 = distance2 + d1 + d2 + d3

                print(str(i) + " " + str(j) + " " + str(k) + " ", end=" ")
                err = (distance2 - value) / len(path)
                print("误差" + str(err), end=" ")

                e = (distance2 - value) / value
                print("相对误差率" + str(e))
                print()

    return


# 对不同状态的数据处理
def fn(i, j, k, classfilter, slot, path):
    re = []
    for p in path:
        # 各状态下的轨迹
        resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward = classifer(classfilter,
                                                                                              slot, p)
        # #除去静止状态后，在拟合处理数据的轨迹的距离
        d1 = countdistance(deal(i, resultSeqUnrealized))
        d2 = countdistance(deal(j, resultSeqBackMove))
        d3 = countdistance(deal(k, resultSeqforward))
        distance2 = d1 + d2 + d3
        re.append(distance2)
    return re


def draw2(title, knn_measure, svm_measure, mlp_measure, trueValue):
    plt.figure(title)
    x = [i for i in range(1, 13)]
    plt.xlabel('轨迹序号', fontproperties='SimHei', fontsize=10)
    plt.ylabel('轨迹长度(米)', fontproperties='SimHei', fontsize=10)
    plt.title(title)

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    plt.plot(x, knn_measure, marker='+', linestyle='--', label='KNN')  # 计算值
    plt.plot(x, svm_measure, marker='*', linestyle='-.', label='SVM')  # 计算值
    plt.plot(x, mlp_measure, marker='p', linestyle=':', label='MLP')  # 计算值

    plt.plot(x, trueValue, marker='o', linestyle='-', label='真实距离')  # 真值
    plt.legend(loc='lower right')  # label生效
    plt.xticks(x)
    plt.show()


# value:真实值
# 使用三种方法分别处理除静止之外的数据
def main2(index, slot, other=15):
    dirPath = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\data"
    path = [dirPath + '\\' + x for x in os.listdir(dirPath)]

    classfilter = ""
    if index == 1:
        classfilter = getKnnClassifer(slot, other)
    elif index == 2:
        classfilter = getSvmClassifer(slot)
    elif index == 3:
        classfilter = getMplClassifer(slot)

    i = 1
    j = 2
    k = 1
    re = fn(i, j, k, classfilter, slot, path)

    return re


# 读取真值
def getvalue(path):
    f = open(path, "r")  # 源文件
    result = []
    for s in f.readlines():
        num = float(s.split("  ")[1][:-1])
        result.append(num)
    return result


# 求和
def distanceofseq(value):
    val_sum = 0
    for i in range(len(value)):
        val_sum += float(value[i])
    return val_sum


def SD(measure, value):
    d = 0  # 标准差
    err = 0  # 平均误差
    sum_value = 0
    sum_measure = 0
    for i in range(len(value)):
        d = d + (value[i] - measure[i]) ** 2
        err += measure[i] - value[i]
        sum_value += value[i]
        sum_measure += measure[i]

    d = np.sqrt(d) / len(value)
    err = err / len(value)
    ff = (sum_measure - sum_value) / sum_value

    print("误差： " + str(err))
    print("相对误差率： " + str(ff))
    print("标准差： " + str(d))
    print()


if __name__ == "__main__":
    path = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\truevalue.txt"
    value = getvalue(path)  # 真值
    knn_measure = main2(1, 35, other=15)
    svm_measure = main2(2, 35, other=15)
    mlp_measure = main2(3, 35, other=15)

    print("KNN")
    getMae(knn_measure, value)
    Mape(knn_measure, value)
    Rmse(knn_measure, value)

    print("SVM")
    getMae(svm_measure, value)
    Mape(svm_measure, value)
    Rmse(svm_measure, value)

    print("MLP")
    getMae(mlp_measure, value)
    Mape(mlp_measure, value)
    Rmse(mlp_measure, value)
