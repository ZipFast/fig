import math


# 误差绝对值 的平均值
def getMae(mearsure, value):
    sum = 0.0

    for i in range(len(value)):
        sum += abs(float(mearsure[i]) - float(value[i]))

    n = len(value)
    sum = sum / n
    print("平均绝对值误差：" + str(sum))
    print()


# 平均相对误差
def Mape(mearsure, value):
    sum = 0.0
    for i in range(1,len(value)):
        sum += abs((float(mearsure[i]) - float(value[i])) / float(value[i]))
    n = len(value)-1
    sum = sum / n
    print("平均相对误差：" + str(sum))
    print()


def Rmse(mearsure, value):
    sum = 0.0
    for i in range(len(value)):
        sum += math.pow(float(mearsure[i]) - float(value[i]), 2)
    n = len(value)
    sum = math.sqrt(sum / n)
    print("均方根差：" + str(sum))
    print()
