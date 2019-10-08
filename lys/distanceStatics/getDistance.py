import numpy as np

import datetime

from classifier.mlpclassifier import getMplClassifer
from classifier.mysvm1 import getSvmClassifer
from classifier.sklearnknn import getKnnClassifer
from dataprocess1.extractcharact import getExtract
from filter.examples.sine_wave import kalmanfilter
import matplotlib.pyplot as plt

from filter.movinAverage import MoveAverage
from filter.sampleFilter import last_square, ployfit, least_sq

plt.rcParams['font.family'] = 'STSong'


# 预处理数据
def datapre():
    pre = 'E:\\datacollect\\test\\test2019-01-19.txt'  # 源数据
    result = 'E:\\datacollect\\test\\test.txt'  # 预处理后数据格式(行)： x y  z time

    f = open(pre, "r")  # 源文件
    fwrit1 = open(result, "w")
    for s in f.readlines():  # 按行读取
        if len(s) == 1:
            pass
        else:
            s = s[:-1]  # s格式   2068 card_x:7.3108 card_y:1.41625 card_z:1.5 2019-01-10 20:12:48\n
            tem = s.split()
            re = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\t" + tem[
                -2] + "\t" + tem[-1] + "\n"
            if tem[0] == "2068":
                fwrit1.write(re)

    f.close()
    fwrit1.close()


# 计算子序列的距离
def getdistanceOfSubSequence(seq):
    th = 1.5  # 阈值
    dist = 0.0
    if len(seq) <= 1:  # 只有一个点时，距离为零
        return dist
    p1 = seq[0]
    for value in seq[1:]:
        p2 = value
        d = np.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)
        if d < th:  # 相邻两点间距离不能太大
            dist += d
        p1 = p2
    return dist


# 卡尔曼滤波，对剩下的数据滤波
def AfterKalmanFilter(list):
    x_vale = []
    y_vale = []
    for item in list:
        x_vale.append(item[0])
        y_vale.append(item[1])
    x_vale = kalmanfilter(x_vale, len(x_vale))

    aftterFilter = [[x_vale[i], y_vale[i]] for i in range(1, min(len(x_vale), len(y_vale)))]

    return aftterFilter


# 参数顺序对应 ：原始序列   静止   无意识运动    往返运动   前行运动
def getResult(originalSeq, resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward):
    # # 滤波
    # resultSeqUnrealized=AfterKalmanFilter(resultSeqUnrealized)
    resultSeqBackMove = AfterKalmanFilter(resultSeqBackMove)
    # resultSeqforward = AfterKalmanFilter(resultSeqforward)

    # 滑动均值拟合

    # print("去除静止数据后，滑动窗口拟合")
    # @参数@2 滑动窗口大小

    # avgObject=MoveAverage(resultSeqUnrealized,4)
    # resultSeqUnrealized=avgObject.smooth()
    #
    # avgObject.seq=resultSeqBackMove
    # resultSeqBackMove = avgObject.smooth()
    #
    # avgObject.seq=resultSeqforward
    # resultSeqforward = avgObject.smooth()

    # #最小二乘  多项式拟合
    originalSeq = least_sq(originalSeq)
    # resultSeqUnrealized= least_sq(resultSeqUnrealized)
    # resultSeqBackMove = least_sq(resultSeqBackMove)
    # resultSeqforward = least_sq(resultSeqforward)

    originalDistance = getdistanceOfSubSequence(originalSeq)
    staticDistance = getdistanceOfSubSequence(resultSeqStatic)

    unrealizedDistance = getdistanceOfSubSequence(resultSeqUnrealized)
    rollbackDistance = getdistanceOfSubSequence(resultSeqBackMove)
    movingDistance = getdistanceOfSubSequence(resultSeqforward)

    # 为使用滤波
    print("原始距离" + str(originalDistance))
    print("静止" + str(staticDistance))
    print("晃动" + str(unrealizedDistance))
    print("往返" + str(rollbackDistance))
    print("前行" + str(movingDistance))
    return unrealizedDistance + rollbackDistance + movingDistance
    # return originalDistance


# filterStyle：分类器类型
# slop:数据划分间隔
def countDistanceOfTatol(filterStyle=1, slop=15):
    # 数据预处理（）
    datapre()
    # 分类器类型
    classfilter = ""
    if filterStyle == 1:
        classfilter = getKnnClassifer(slop)
    elif filterStyle == 2:
        classfilter = getSvmClassifer(slop)
    else:
        classfilter = getMplClassifer(slop)

    origenal = []  # 原始坐标序列（不做任何处理），计算原始距离
    resultSeqStatic = []  # 去掉静止子序列后的数据
    resultSeqUnrealized = []  # 去掉静止子序列后的数据
    resultSeqBackMove = []  # 去掉静止子序列后的数据
    resultSeqforward = []  # 去掉静止子序列后的数据
    # 运动状态分析，与剔除
    path = 'E:\\datacollect\\test\\test.txt'  # 移动端的定位坐标
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
                seq = s.split("\t")  # 按"\t"切分字符
                if flag == False:
                    start = seq[3] + "\t" + seq[4]  # 本子数据序列第一个点采集时间
                    flag = True
                now = seq[3] + "\t" + seq[4]  # 当前点采集时间
                subt = (datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(start,
                                                                                                          "%Y-%m-%d %H:%M:%S")).seconds
                # 逐个获取序列
                if subt > slop:

                    if len(slopSequen) > 0:
                        vector = getExtract(slopSequen)  # 计算序列特征[[v1,v2,v3]]
                        state = classfilter.predict(vector)[0]  # 判断当前序列的运动状态
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
                    temp.append(float(seq[0]))
                    temp.append(float(seq[1]))
                    slopSequen.append(temp)
                    origenal.append(temp)
                    start = now
                    pass
                else:
                    temp = []
                    temp.append(float(seq[0]))
                    temp.append(float(seq[1]))
                    slopSequen.append(temp)
                    origenal.append(temp)

    # 路程
    dist = getResult(origenal, resultSeqStatic, resultSeqUnrealized, resultSeqBackMove, resultSeqforward)
    return dist


def draw(title, index, result):
    fig = plt.figure(index)
    plt.xlabel('x间隔', fontproperties='SimHei', fontsize=20)
    plt.ylabel('y数量', fontproperties='SimHei', fontsize=20)
    plt.title(title)

    x = [i for i in range(5, 40, 2)]
    y_knn, y_svm, y_mpl = [], [], []
    for item in result:
        y_knn.append(item[0])
        y_svm.append(item[1])
        y_mpl.append(item[2])

    t = []  # 真值
    for i in range(len(x)):
        t.append(1123.37)

    plt.plot(x, y_knn, marker='x', color='b', label='knn')
    plt.plot(x, y_svm, linestyle='-', marker='o', color='r', label='svm')
    plt.plot(x, y_mpl, marker='*', color='g', label='mpl')
    plt.plot(x, t, marker=".", color='yellow', label='true')
    plt.legend(loc='lower right')  # label生效

    plt.savefig("E:\\datacollect\\result\\" + title + "_" + str(index) + ".png")
    plt.close(fig)


def drawDetail():
    # ss=['knn' ,'svm' ,'mpl']
    styl = [1, 2, 3]

    index = 1  # 计数
    title = u"10"
    result = []  # 保存各个时间分隔，计算得到的距离，用于绘图分析
    num = 1
    for i in range(5, 40, 2):  # 从5s开始到40s,步伐为2
        temp = []
        for s in styl:  # 测试该间隔下，使用不同分类模型 ，得到的距离
            distance = countDistanceOfTatol(s, i)
            print(num, end="  ")
            num += 1
            print(distance)
            temp.append(distance)
        result.append(temp)
    draw(title, index, result)


if __name__ == '__main__':
    temp = []
    for i in range(20, 30, 1):  # 从5s开始到40s,步伐为2
        distance = countDistanceOfTatol(1, i)
        print(i, end="  ")
        print(distance)
        temp.append(distance)
    print(temp)
