import numpy as np
class MoveAverage():
    """对数据序列进行平滑拟合—滑动均值拟合算法"""

    # seq为原始数据，n为窗口大小;并返回str='avgX  avgY'
    def __init__(self, seq, n):
        self.seq = seq  # 数组中item=x y z datetime
        self.n = n

    # 计算数组tem中x,y均值
    def count(self, tem):
        x_vag = 0.0
        y_avg = 0.0
        for v in tem:
            x_vag += float(v[0])  # 累加
            y_avg += float(v[1])

        num = float(self.n)
        st = [x_vag / num, y_avg / num]  # 计算均值
        return st

    # 返回拟合后的数据
    def smooth(self):
        result = []  # 拟合的结果
        # 数据长度小于 窗口大小，直接返回
        if len(self.seq) <= self.n:
            return self.seq
        win = self.seq[:self.n]  # win数组为滑动窗口，取原始数据的n个初始化

        # 从第n个点开始平滑
        for value in self.seq[self.n:]:
            result.append(self.count(win))  # 取当前窗口中个均值作为输出
            del win[0]  # 删除窗口第一个点
            win.append(value)  # 在窗口尾部添加一个点

        # 最后将窗口中的值加上
        for v in win[1:]:
            result.append(v)

        return np.asarray(result)
