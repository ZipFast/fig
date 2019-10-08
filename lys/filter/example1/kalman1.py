# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

# by Andrew D. Straw


# 对x进行估计
#采用正态分布产生x的测量值

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)

def filter(x_value):

    # 初始参数
    n_iter = len(x_value)  # 迭代50次
    sz = (n_iter,)  # 数组的大小
    # x = -0.37727 # 真值(typo in example at top of p. 13 calls this z)

    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    # 参数的意义为：
    # 　　loc:float
    # 　　概率分布的均值，对应着整个分布的中心center
    # 　　scale:float
    # 　　概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高
    # 　　size:int or tuple of ints
    # 　　输出的shape，默认为None，只输出一个值
    # 　　我们更经常会用到np.random.randn(size)所谓标准正太分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)
    # z = np.random.normal(x,0.1,size=sz) # 观察值(normal about x, sigma=0.1)

    rnd = np.random.RandomState(0)  # 为随机数生成器，确保生成一致的随机数

    # generate a noisy sine wave to act as our fake observations
    # n_timesteps = 200
    # x_axis = np.linspace(0, 5 * np.pi, n_timesteps)  # 在指定的间隔内返回均匀间隔的数字。
    # z = 20 * (np.sin(x_axis) + 0.5 * rnd.randn(n_timesteps))  # 计算正弦值并加入噪声，然后放大20倍
    # x = 20 * np.sin(x_axis)

    Q = 2e-5  # 过程方差
    # 为数组分配空间
    xhat = np.zeros(sz)  # 后验估计x
    P = np.zeros(sz)  # 后验误差估计
    xhatminus = np.zeros(sz)  # 先验估计x
    Pminus = np.zeros(sz)  # 先验估计误差
    K = np.zeros(sz)  # 增益或混合系数

    R = 0.05 ** 2  # 测量方差估计，变化见效果

    # 开始预测
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (x_value[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]


    return np.asarray(xhat)

