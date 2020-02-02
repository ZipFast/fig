# 拟合滤波处理轨迹后，统计运动距离

import os
import numpy as np



# 卡尔曼滤波
from drawimg.resultOfClassifier import draw2, draw4
from evaluate import getMae, Mape, Rmse
from filter.examples.sine_wave import kalmanfilter
from filter.movinAverage import MoveAverage
from filter.sampleFilter import last_polyfit


def preDataOfKalman(path):
    f = open(path, "r")  # 源文件
    sequence = []
    x_value = []
    y_value = []
    for s in f.readlines():
        if len(s) == 1:  # 是否空行
            continue
        else:
            s = s[:-1]  # 去除行尾的换行符
            tem = s.split()  # 按空格切分字符串
            if tem[0] == "2068":
                x_value.append(float(tem[1].split(':')[1]))
                y_value.append(float(tem[2].split(':')[1]))
    f.close()

    x_value = kalmanfilter(x_value, len(x_value))  # 滤波

    length = min(len(x_value), len(y_value))  # 数组长度
    sequence = [[x_value[i], y_value[i]] for i in range(0, length)]
    return sequence


# 最小二次法曲线拟合
def preDataOfleastSquare(path):
    f = open(path, "r")  # 源文件
    sequence = []
    for s in f.readlines():
        if len(s) == 1:
            continue
        else:
            s = s[:-1]  # 去除行尾的换行符
            tem = s.split()  # 按空格切分字符串
            if tem[0] == "2068":
                x_value = float(tem[1].split(':')[1])
                y_value = float(tem[2].split(':')[1])
                sequence.append([x_value, y_value])
    f.close()
    result = last_polyfit(sequence)
    return result


# 滑动均值
def preDataOfMovingAverage(path, n=4):
    f = open(path, "r")  # 源文件
    sequence = []  # 坐标序列

    for s in f.readlines():
        if len(s) == 1:
            continue
        else:
            s = s[:-1]  # 去除行尾的换行符
            tem = s.split()  # 按空格切分字符串
            if tem[0] == "2068":
                x_value = float(tem[1].split(':')[1])
                y_value = float(tem[2].split(':')[1])
                sequence.append([x_value, y_value])
    f.close()

    obj = MoveAverage(sequence, n)  # 创建对象
    result = obj.smooth()  # 平滑数据

    return result


# 遍历文件夹中的所有文件
def count(file_dir):
    path = [file_dir + '\\' + x for x in os.listdir(file_dir)]
    Kresult = []  # 卡尔曼滤波处理后，计算的结果
    Lresult = []  # 最小二乘法
    Aresult = []  # 滑动均值
    for p in path:
        distance = 0.0
        distance1 = 0.0
        distance2 = 0.0

        seq = preDataOfKalman(p)
        distance = float("%.2f" % countdistance(seq))
        Kresult.append(distance)

        seq1 = preDataOfleastSquare(p)
        distance1 = float("%.2f" % countdistance(seq1))
        Lresult.append(distance1)

        seq2 = preDataOfMovingAverage(p, 11)
        distance2 = float("%.2f" % countdistance(seq2))
        Aresult.append(distance2)
    return Kresult, Lresult, Aresult


# 计算序列对应的距离
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
            distance += d
            p1 = p2
        return distance


# 读取真值
def getvalue(path):
    f = open(path, "r")  # 源文件
    result = []
    for s in f.readlines():
        num = float(s.split("  ")[1][:-1])
        result.append(num)
    return result







if __name__ == "__main__":
    dirPath = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\data"
    kf, least, avgWin = count(dirPath)  # 计算值

    path = "D:\\项目\\廖煜胜-论文\\论文\\数据与程序\\datacollect\\test\\truevalue.txt"
    value = getvalue(path)  # 真值

    print("卡尔曼")
    getMae(kf, value)
    Mape(kf, value)
    Rmse(kf, value)

    print("最小二乘")
    getMae(least, value)
    Mape(least, value)
    Rmse(least, value)

    print("滑动均值")
    getMae(avgWin, value)
    Mape(avgWin, value)
    Rmse(avgWin, value)