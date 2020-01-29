'''
@Descripttion: 
@version: 
@Author: zhangwenjum
@Date: 2020-01-22 14:35:03
@LastEditors  : zhangwenjum
@LastEditTime : 2020-01-22 14:45:33
'''
import matplotlib.pyplot as plt 
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
t = [10, 20, 30, 40, 50]
acc = [91.3867,  90.8414, 91.7429, 97.4448, 93.3361]
plt.ylim((60, 100))
plt.plot(t, acc)
plt.xlabel("时间间隔 t")
plt.ylabel("查准率 acc") 
plt.title("查准率随时间间隔变化图")
plt.legend()
plt.show()