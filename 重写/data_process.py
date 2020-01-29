'''
@Descripttion: 
@version: 
@Author: zhangwenjum
@Date: 2020-01-28 10:26:32
@LastEditors  : zhangwenjum
@LastEditTime : 2020-01-28 14:42:54
'''
"""
数据预处理：原始数据格式为(x, y, z, time); 表示坐标点的三维坐标和收集时间
"""

import os
import numpy as np

def pre(source, distance):
    """ 对源文件预处理
    args:
        source: 源文件
        distance: 处理后的三维坐标文件
        
    return:
        None 
    """
    f = open(source, "r")
    fwrit = open(distance, "a")
    for line in f.readlines():
        if len(line) == 1:
            fwrit.write(line)
        else:
            line = line[:-1]
            tem = line.split()
            re = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\t" + tem[-2] + "\t" + tem[-1] + "\n"
            if tem[0] == "2068":
                """
                读写2068卡的数据
                """
                fwrit.write(re)
    f.close()
    fwrit.close()

def file_name(file_dir, target):
    """ 对给定目录下的所有文件进行预处理
    Args:
        file_dir: 文件目录
        target: 文件名
    Return:
        None 
    """
    path = [file_dir + '\\' + x for x in os.listdir(file_dirs)]
    for p in path:
        if not os.path.isdir(p):
            pre(p, target)

def split_data(splot)
    """ 按照给定的时间间隔切分数据
    Args:
        splot: Integer 代表时间间隔
    Return:
        splotre: numpy array of data 
        label: numpy array of data labell
    """
    """
       下面四种状态分别代表静止状态，无意识运动，小范围来回运动
    大范围来回运动
    """
    state = "D:\\fig\\data\\pre2068Static.txt"
    unrealize = "D:\\fig\\data\\pre2068Unrealize.txt"
    Sactive = "D:\\fig\\data\\pre2068Little.txt"
    Mactive = "D:\\fig\\data\\pre2068LargeMove.txt"

    files = [state, unrealize, Sactive, Mactive]
    mask = [0., 1., 2., 3.] 
    splotre = [] 
    label = [] 
    for index, file in enumerate(files):
        f = open(file, "r")
        mk = mask[index]

        flag = False
        start = ''
        obj = [] 
        for s in f.readlines():
            if len(s) <= 1:
                continue
            else:
                s = s[:-1]
                seq = s.split("\t")
                if flag == False:
                    start = 0 
                    now = 0 
                    flag = True
                now = now + 1 
                subt = now - start 
                obj.append(np.asarray(seq[:3], dtype='float64'))
            if subt > splot:
                splotre.append(np.asarray(obj))
                label.append(mk)
                obj.clear()
                flag = False
    splotre = np.asarray(splotre)
    label = np.asarray(label)