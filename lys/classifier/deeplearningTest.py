import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
"""
提取每个序列的特征： 局部距离（特征1）、相异点数量（特征2），均方差（特征3）
类别={'静止'：1，’小运动‘：2，’大运动‘：3 }
格式：
特征1，特征2，特征3，类别
"""

import datetime
import numpy as np
import math

np.set_printoptions(suppress=True)  # 去除科学计数法


# 加载数据训练和测试模型
# 按splot间隔划分子序列，并计算特征
# 返回一个二维数组（每行表示一个序列的特征向量）和一个一位数组（类别）
def normData(splot):
    # 数据源
    state = "E:\\datacollect\\pre\\pre2068.txt"  # (静止)
    unrealize = "E:\\datacollect\\pre\\pre2068unrealize.txt"  # 运动
    Sactive = "E:\\datacollect\\pre\\pre2068movelittle.txt"  # 小范围运动
    Mactive = "E:\\datacollect\\pre\\pre2068active.txt"  # 运动

    files = [state, unrealize, Sactive, Mactive]
    mask = [1.0, 2.0, 3.0, 4.0]  # 1.0表示静止，2.0表示无意识动，3.0表示小动，4.0表示动

    # mask = [1.0, 1.0, 2.0, 3.0]

    result = []  # 特征向量集合
    lable = []  # 标签集合
    splotre = []  # 一个序列
    # write=open(predata,"w")
    for index, file in enumerate(files):
        f = open(file, "r")  # 源文件
        mk = mask[index]  # 类别

        flag = False
        start = ''  # 记录每个序列的起始时间
        for s in f.readlines():
            if len(s) <= 1:  # 跳过空行
                continue
            else:
                s = s[:-1]  # 去掉换行符
                seq = s.split("\t")  # 按"\t"切分字符
                if flag == False:
                    start = seq[3] + "\t" + seq[4]  # 本数据序列第一个点采集时间
                    flag = True
                now = seq[3] + "\t" + seq[4]  # 当前点采集时间
                subt = (datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(start,
                                                                                                          "%Y-%m-%d %H:%M:%S")).seconds

                # 计算一个系列的特征：局部距离，相异点数量，均方差
                if subt > splot:
                    #
                    arr = np.asarray(splotre)  # 转成数组

                    # 求序列中x和y的平均值
                    xavg = 0.0
                    yavg = 0.0

                    # 计算局部距离(特征1)
                    distance = 0
                    n = arr.shape[0]  # 序列中点的数目
                    if (n >= 2):  # 坐标点数大于2
                        p0 = arr[0]  # 取第一个点
                        xavg += p0[0]
                        yavg += p0[1]
                        for p in arr[1:]:
                            d = math.sqrt((p0[0] - p[0]) ** 2 + (p0[1] - p[1]) ** 2)
                            if d <= 2.0:
                                distance += d
                            p0 = p
                            xavg += p[0]
                            yavg += p[1]

                    xavg = xavg / float(n)
                    yavg = yavg / float(n)

                    # 计算相异点数目
                    # 计算方差(各点到质心距离的平方和的平均数)
                    s = set()  # 将坐标点以元组的形式保存到集合中，利用其互异性，去掉相同点
                    fp = 0.0
                    for p in arr:
                        d0 = (p[0] - xavg) ** 2 + (p[1] - yavg) ** 2  # 各点到质心的距离的平方
                        fp += d0
                        s.add((p[0], p[1]))
                    fp = fp / float(n)  # 特征2
                    count = float(len(s))  # 特征3

                    # 保留三位小数
                    distance = float('%.3f' % distance)
                    fp = float('%.3f' % fp)

                    # #保存到磁盘
                    #  out=str(distance)+" "+str(fp)+" "+str(count)+" "+str(mk)
                    #  write.write(out+'\n')

                    tem = []  # 三个特征构成向量  p=(distance,fp,count)
                    tem.append(distance)
                    tem.append(fp)
                    tem.append(count)

                    # tem.append(mk)
                    result.append(tem)  # 特征向量序列
                    lable.append(mk)  # 特征向量 （子序列）对应的类型

                    # 设置下个序列的起始时间，和保存第一个点
                    splotre = []  # 清空上一个序列数据
                    start = now
                    temp = []
                    temp.append(float(seq[0]))
                    temp.append(float(seq[1]))
                    splotre.append(temp)

                    pass
                else:
                    temp = []  # 将一个坐标点一数组形式保存
                    temp.append(float(seq[0]))  # x坐标
                    temp.append(float(seq[1]))  # y坐标
                    splotre.append(temp)  # 坐标序列
        f.close()

    return np.array(result), np.asarray(lable).reshape(len(lable), 1)


# 计算一个序列的局部距离、均方差和互异点数
def getExtract(seq):
    arr = np.asarray(seq)  # 转成二位数组
    # 求序列中x和y的平均值
    xavg = 0.0
    yavg = 0.0

    # 计算局部距离(特征1)
    distance = 0
    n = arr.shape[0]  # 序列中点的数目
    if (n >= 2):  # 坐标点数大于2
        p0 = arr[0]  # 取第一个点
        xavg += p0[0]
        yavg += p0[1]
        for p in arr[1:]:
            d = math.sqrt((p0[0] - p[0]) ** 2 + (p0[1] - p[1]) ** 2)
            if d <= 2.0:
                distance += d
            p0 = p
            xavg += p[0]
            yavg += p[1]

        xavg = xavg / float(n)
        yavg = yavg / float(n)

    # 计算相异点数目
    # 计算方差(各点到质心距离的平方和的平均数)
    s = set()  # 将坐标点以元组的形式保存到集合中，利用其互异性，去掉相同点
    fp = 0.0
    for p in arr:
        d0 = (p[0] - xavg) ** 2 + (p[1] - yavg) ** 2  # 各点到质心的距离的平方
        fp += d0
        s.add((p[0], p[1]))  # 将每个点一元组形式保存到集合中，利用集合互异性剔除重复的点
    fp = fp / float(n)  # 特征2
    count = float(len(s))  # 特征3

    # 保留三位小数
    distance = float('%.3f' % distance)
    fp = float('%.3f' % fp)

    tem = [distance, fp, count]
    tem = [tem]
    return np.asarray(tem)

torch.manual_seed(1)

epoches = 1  # 训练整批数据的次数
batch_size = 500  # 每次训练的样本数
lr = 0.001

def getCNN(slot):
    np.set_printoptions(suppress=True)
    x, lable = normData(slot)
    x = torch.from_numpy(x)
    lable = torch.from_numpy(lable)
    net = Net(n_feature=3, n_hidden=10, n_output=4)  # 几个类别就几个 output
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 传入 net 的所有参数, 学习率
    # 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
    # 但是预测值是2D tensor (batch, n_classes)
    loss_func = torch.nn.CrossEntropyLoss()
    plt.ion()  # 画图
    plt.show()
    for t in range(100):
        out = net(x)  # 喂给 net 训练数据 x, 输出分析值

        loss = loss_func(out, lable)  # 计算两者的误差

        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        if t % 2 == 0:
            plt.cla()
            # 过了一道 softmax 的激励函数后的最大概率才是预测值
            prediction = torch.max(F.softmax(out), 1)[1]
            pred_y = prediction.data.numpy().squeeze()
            target_y = lable.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.out(x)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x
if __name__ == '__main__':
    getCNN(10.)


