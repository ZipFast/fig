import os
import numpy as np
from drawimg.resultOfClassifier import draw1


# 直接计算各个轨迹的距离
# 误差

# 为: x   y   z  time     ；表示一个坐标点的三个坐标分量和 采集时间 ，使用空格符分隔
from evaluate import getMae, Mape, Rmse


def preData(path):
    f = open(path, "r")  # 源文件
    sequence = []
    for s in f.readlines():
        if len(s) == 1:
            continue
        else:
            s = s[:-1]
            tem = s.split()
            line = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\t" + tem[
                -2] + "\t" + tem[-1]
            if tem[0] == "2068":
                sequence.append(line)
    f.close()
    return sequence


# 遍历文件夹中的所有文件
def countDistance(file_dir):
    path = [file_dir + '\\' + x for x in os.listdir(file_dir)]
    result = []
    for p in path:
        distance = 0.0
        seq = preData(p)

        # 计算一个序列的运动距离
        if len(seq) <= 1:  # 只有一个点时，距离为零
            result.append(distance)
        else:
            p1 = seq[0].split("\t")
            for value in seq[1:]:
                p2 = value.split("\t")
                d = np.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)
                distance += d
                p1 = p2
            result.append(distance)
    return result


def getvalue(path):
    f = open(path, "r")  # 源文件
    result = []
    for s in f.readlines():
        num = float(s.split("  ")[1][:-1])
        result.append(num)
    return result


if __name__ == "__main__":
    dirPath = "E:\\datacollect\\test\\data"
    measure = countDistance(dirPath)

    path = "E:\\datacollect\\test\\truevalue.txt"
    value = getvalue(path)

    print("滑动均值")
    getMae(measure, value)
    Mape(measure, value)
    Rmse(measure, value)