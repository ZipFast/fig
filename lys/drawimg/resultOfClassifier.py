# 将分类准确率绘图

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Simhei']


# lable_x 时间range(5,60,5)
# label_y 准确率
# style 分类器类型   保存路径
def draw(style, title, label_x, label_y):
    plt.figure(title)

    plt.xlabel('x 时间间隔', fontproperties='SimHei', fontsize=10)
    plt.ylabel('y 分类精度', fontproperties='SimHei', fontsize=10)
    plt.title(title)

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    plt.plot(label_x, label_y, marker='+')

    if style == 1:
        plt.savefig("E:\\datacollect\\picture\\experiment\\knn\\" + str(title) + ".png")
    elif style == 2:
        plt.savefig("E:\\datacollect\\picture\\experiment\\svm\\" + str(title) + ".png")
    else:
        plt.savefig("E:\\datacollect\\picture\\experiment\\mpl\\" + str(title) + ".png")


# 直接统计距离
def draw1(title, measure, trueValue):
    plt.figure(title)
    x = [i for i in range(1, 13)]
    plt.xlabel('轨迹序号', fontproperties='SimHei', fontsize=10)
    plt.ylabel('轨迹长度（米）', fontproperties='SimHei', fontsize=10)
    plt.title(title)

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)
    plt.xticks(x)
    for i in range(1, len(x)):
        d = ((measure[i] - trueValue[i]) / trueValue[i]) * 100
        text = float("%.2f" % d)
        plt.text(x[i] - 0.4, measure[i] + 30, str(text) + "%", fontdict={'size': '10'})

    plt.plot(x, measure, marker='*', linestyle='-.', label='直接计算')  # 计算值
    plt.plot(x, trueValue, marker='o', linestyle='-', label='真实值')  # 真值
    plt.legend(loc='lower right')  # label生效

    plt.show()


def draw4(title, measure, trueValue, label):
    plt.figure(title)
    x = [i for i in range(1, 13)]
    plt.xlabel('轨迹序号', fontproperties='SimHei', fontsize=10)
    plt.ylabel('轨迹长度（米）', fontproperties='SimHei', fontsize=10)
    plt.title("计算值与真实值对比")

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)
    plt.xticks(x)
    for i in range(1, len(x)):
        d = ((measure[i] - trueValue[i]) / trueValue[i]) * 100
        text = float("%.2f" % d)
        plt.text(x[i] - 0.4, measure[i] + 30, str(text) + "%", fontdict={'size': '10'})

    plt.plot(x, measure, marker='*', linestyle='-.', label=label)  # 计算值
    plt.plot(x, trueValue, marker='o', linestyle='-', label='真实值')  # 真值
    plt.legend(loc='lower right')  # label生效
    plt.savefig("E:\\datacollect\\picture\\experiment\\distance\\" + str(title) + ".png")


def draw2(title, measureOfKalman, measureOfLeast, measureOfAvg, trueValue):
    plt.figure(title)
    x = [i for i in range(1, 13)]
    plt.xlabel('轨迹序号', fontproperties='SimHei', fontsize=10)
    plt.ylabel('轨迹长度(米)', fontproperties='SimHei', fontsize=10)
    plt.title(title)

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    plt.plot(x, measureOfKalman, marker='+', linestyle='--', label='卡尔曼滤波')  # 计算值
    plt.plot(x, measureOfLeast, marker='*', linestyle='-.', label='最小二乘法')  # 计算值
    plt.plot(x, measureOfAvg, marker='p', linestyle=':', label='滑动均值')  # 计算值

    plt.plot(x, trueValue, marker='o', linestyle='-', label='真实距离')  # 真值
    plt.legend(loc='lower right')  # label生效
    plt.xticks(x)
    plt.show()


def draw3(title, measure4, measure3, trueValue):
    plt.figure(title)
    x = [i for i in range(1, 12)]
    plt.xlabel('x ', fontproperties='SimHei', fontsize=10)
    plt.ylabel('y (米)', fontproperties='SimHei', fontsize=10)
    plt.title(title)

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    plt.plot(x, measure4, marker='+', linestyle='--', color='green', label='去除静止状态')  # 计算值
    plt.plot(x, measure3, marker='*', linestyle='-.', color='blue', label='去除静止和无意识运动状态')  # 计算值

    plt.plot(x, trueValue, marker='o', linestyle='-', color='r', label='true')  # 真值
    plt.legend(loc='lower right')  # label生效

    # 平均绝对误差
    kferror = avgError(measure4, trueValue)
    Leasterror = avgError(measure3, trueValue)

    plt.text(8, 3250, "average error: " + str(kferror), size=11, family="fantasy", color="green", style="italic")
    plt.text(8, 3000, "average error: " + str(Leasterror), size=11, family="fantasy", color="blue", style="italic")

    plt.show()


def avgError(measure, trueValue):
    sum = 0
    for i in range(len(measure)):
        sum += measure[i] - trueValue[i]
    sum = float('%.2f' % (sum / len(measure)))
    return sum
