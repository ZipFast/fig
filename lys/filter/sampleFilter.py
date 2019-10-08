from scipy.optimize import curve_fit, leastsq
import numpy as np


#

# 最小二乘—指数拟合

# 自定义指数函数（高斯、多项式....）
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c


# 最小二乘-多项式

### 需要拟合的函数func及误差error ###
def func_least(x, a, b, c, d):
    return a * (x ** 3) + b * (x ** 2) + c * x + d


def last_polyfit(sequen):
    if len(sequen) <= 1:
        return sequen
    xdata, ydata = [], []
    for item in sequen:
        xdata.append(item[0])
        ydata.append(item[1])
    popt, pcov = curve_fit(func_least, xdata, ydata, maxfev=10000)  #
    # popt数组中，三个值分别是待求参数a,b,c,d
    y2 = [func_least(i, popt[0], popt[1], popt[2], popt[3]) for i in xdata]
    return [[xdata[i], y2[i]] for i in range(len(y2))]
