import matplotlib.pyplot as plt
import numpy as np
import datetime

# 只绘制运动状态的数据集
# 按不同的时间间隔将 数据绘图

plt.rcParams['font.family'] = 'STSong'
plt.rcParams['font.size'] = 11

# plt.rcParams['savefig.dpi'] = 28  # 图片像素
# plt.rcParams['figure.dpi'] = 28  # 分辨率

timeSeq = 30


def draw(x, y, name, num):
    n = 0
    # 判断不相同的坐标点数量
    pzip = zip(x, y)  # 返回对象
    pzip = list(pzip)
    s = set()  # 利用集合特性去除相同点
    for e in pzip:
        s.add(e)
    n = len(s)  # 不同点的数目

    fig = plt.figure(name)
    plt.xlabel('横坐标(米)', fontproperties='SimHei', fontsize=15)
    plt.ylabel('纵坐标(米)', fontproperties='SimHei', fontsize=15)
    # plt.title('10秒采集坐标数(共：' + str(num) +"个点，去除相同点后，剩下："+str(n)+"个)")

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    plt.xlim(0.0, 8.0)
    plt.ylim(-1, 4.5)



    # plt.xlim(-3.0, 10)
    # plt.ylim(-3.0, 10)
    #
    # plt.xticks(np.arange(-3.0, 10, 0.5))
    # plt.yticks(np.arange(-3.0, 10, 0.5))

    plt.scatter(x, y, c='blue', alpha=1, s=20, marker='*')  # c='blue'定义为蓝色
    plt.grid(True)
    # plt.show()
    plt.savefig("E:\\datacollect\\picture\\active\\10sactive\\" + str(name) + ".png")
    #plt.savefig("E:\\datacollect\\picture\\imge\\img" + str(count) + "_" + "4.jpg")
    plt.close(fig)


def main():
    t = "E:\\datacollect\\pre\\pre2068active.txt"
    fwrit = open(t, "r")

    x = []
    y = []
    count = 1  # 图编号
    n = 0  # 每个图点的数量
    start = ''
    for s in fwrit.readlines():
        if len(s) == 1:
            continue
        else:
            # 记录第一个点的时间
            seq = s.split("\t")

            if len(x) == 0:
                start = seq[3] + "\t" + seq[4][:-1]  # 本数据序列第一个点采集时间

            now = seq[3] + "\t" + seq[4][:-1]  # 当前点采集时间
            subt = (datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(start,
                                                                                                      "%Y-%m-%d %H:%M:%S")).seconds

            if subt > timeSeq:

                draw(x, y, count, n)
                count += 1

                # 重新保存
                x = []
                y = []
                n = 0
                x.append(float(seq[0]))
                y.append(float(seq[1]))
                n += 1
                start = seq[3] + "\t" + seq[4][:-1]
            else:
                x.append(float(seq[0]))
                y.append(float(seq[1]))
                n += 1

    print(count - 1)
    fwrit.close()


if __name__ == "__main__":
    main()
