import os


##文件数据行为: x   y   z  time     ；表示一个坐标点的三个坐标分量和 采集时间 ，使用空格符分隔

def pre(source, distance, diantance1):
    f = open(source, "r")  # 源文件
    fwrit = open(distance, "a")  # 卡2068
    for s in f.readlines():
        if len(s) == 1:
            fwrit.write(s)
        else:
            s = s[:-1]
            tem = s.split()
            re = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\n" 
            # if tem[0] != "2068":
            #     fwrit1.write(re)
            # else:
            if tem[0] == "2068":
                fwrit.write(re)
    f.close()
    fwrit.close()


def file_name(file_dir, target, target1):
    path = [file_dir + '\\' + x for x in os.listdir(file_dir)]
    for p in path:
        if not os.path.isdir(p):
            pre(p, target, target1)


if __name__ == '__main__':
    # 大幅度运动 原始数据源
    p = "E:\\datacollect\\trian\\active"
    # 数据预处理结果保存路径
    t = "D:\\fig\\data\\pre2068LargeMove.txt"  # 卡2068对应的数据，处理结果
    t1 = "D:\\fig\\data\\preLargeMove.txt"  # 其他卡的处理结果

    file_name(p, t, t1)
    p = "E:\\datacollect\\trian\\little"  # 原始数据源

    # 数据预处理结果保存路径
    t = "D:\\fig\\data\\pre2068Little.txt"  # 卡2068对应的数据，处理结果
    t1 = "D:\\fig\\data\\preLittle.txt"  # 其他卡的处理结果

    file_name(p, t, t1)
    p = "E:\\datacollect\\trian\\static"  # 原始数据源

    # 数据预处理结果保存路径
    t = "D:\\fig\\data\\pre2068Static.txt"  # 卡2068对应的数据，处理结果
    t1 = "D:\\fig\\data\\preStatic.txt"  # 其他卡的处理结果
    file_name(p, t, t1)
    # 无意识运动，如转身，手摆动
    p = "E:\\datacollect\\trian\\unrealize"  # 原始数据源
    # 数据预处理结果保存路径
    t = "D:\\fig\\data\\pre2068Unrealize.txt"  # 卡2068对应的数据，处理结果
    t1 = "D:\\fig\\data\\preUnrealize.txt"  # 其他卡的处理结果径
    file_name(p, t, t1)


