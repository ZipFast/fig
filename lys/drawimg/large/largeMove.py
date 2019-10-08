import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'STSong'
plt.rcParams['font.size'] = 11


# 只绘制运动状态的数据集  60s

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
    plt.xlabel('x轴', fontproperties='SimHei', fontsize=20)
    plt.ylabel('y轴', fontproperties='SimHei', fontsize=20)
    plt.title('60秒采集坐标数(共：' + str(num) + "个点，去除相同点后，剩下：" + str(n) + "个)")

    plt.subplots_adjust(bottom=0.15)  # 图边距t
    plt.subplots_adjust(left=0.15)

    # plt.xlim(0, 6)
    # plt.ylim(0, 6)
    #
    # plt.xticks(np.arange(0, 6.0, 0.5))
    # plt.yticks(np.arange(0, 6.0, 0.5))

    plt.xlim(-3.0, 10)
    plt.ylim(-3.0, 10)

    plt.xticks(np.arange(-3.0, 10, 0.5))
    plt.yticks(np.arange(-3.0, 10, 0.5))

    plt.scatter(x, y, alpha=1, s=20, marker='*')  # c='blue'定义为蓝色
    plt.grid(True)
    # plt.show()
    plt.savefig("E:\\datacollect\\picture\\active\\60sactive\\" + str(name) + ".png")
    plt.close(fig)


def main():
    t = "E:\\datacollect\\pre2068active.txt"
    fwrit = open(t, "r")

    x = []
    y = []
    count = 1  # 图编号
    n = 0  # 每个图点的数量
    for s in fwrit.readlines():
        if len(s) == 1:
            if len(x) == 0:
                continue
            draw(x, y, count, n)
            count += 1
            x = []
            y = []
            n = 0
        else:
            seq = s.split("\t")
            x.append(float(seq[0]))
            y.append(float(seq[1]))
            n += 1

    print(count - 1)
    fwrit.close()


if __name__ == "__main__":
    main()
