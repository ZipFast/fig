import os


# 文件数据行为: x   y   z  time
# 表示一个坐标点的三个坐标分量和 采集时间 ，使用空格符分隔

def pre(source, distance, diantance1):
    f = open(source, "r")  # 源文件
    fwrit = open(distance, "a")

    for s in f.readlines():
        if len(s) == 1:
            fwrit.write(s)
        else:
            s = s[:-1]
            tem = s.split()
            re = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\t" + tem[
                -2] + "\t" + tem[-1] + "\n"
            if tem[0] == "2068":
                fwrit.write(re)
    f.close()
    fwrit.close()


# 遍历文件夹中的所有文件
def file_name(file_dir, target, target1):
    path = [file_dir + '\\' + x for x in os.listdir(file_dir)]
    for p in path:
        if not os.path.isdir(p):
            pre(p, target, target1)


if __name__ == '__main__':
    # 静止
    p = "E:\\datacollect\\trian\\static"  # 原始数据源

    # 数据预处理结果保存路径
    t = "E:\\datacollect\\pre\\pre2068.txt"  # 保存2068标签数据
    t1 = "E:\\datacollect\\pre\\pre.txt"  # 其他卡路径
    file_name(p, t, t1)
