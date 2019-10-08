import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'STSong'


def draw():
    title = u"最小二乘-二次多项式拟合往返序列"

    fig = plt.figure()
    plt.xlabel('x间隔', fontproperties='SimHei', fontsize=20)
    plt.ylabel('y数量', fontproperties='SimHei', fontsize=20)
    plt.title(title)

    # 真值 和不做任何处理
    x = [i for i in range(20, 30, 1)]
    t = []
    for i in range(len(x)):
        t.append(3120.0)

    # 未做处理
    seqoriginal = [3874.4429314067734, 3874.4429314067734, 3874.4429314067734, 3874.4429314067734, 3874.4429314067734,
                   3874.4429314067734, 3874.4429314067734, 3874.4429314067734, 3874.4429314067734, 3874.4429314067734]
    mvg = [3139.6079269167635, 3144.8402923231374, 3214.9320196181598, 3143.1399156365856, 3169.5387395553425,
           3164.756705386748, 3154.2404940842252, 3165.1878822941726, 3127.434825449098, 3126.176167806726]
    plt.plot(x, mvg, linestyle='-', marker='o', color='r', label='拟合处理')
    plt.plot(x, seqoriginal, marker='*', color='g', label='未处理')
    plt.plot(x, t, marker=".", color='yellow', label='true')
    plt.legend(loc='lower right')  # label生效

    plt.savefig("E:\\datacollect\\result\\" + title + ".png")
    plt.close(fig)


if __name__ == "__main__":
    draw()
