import os


# 将2068 的数据更改为6881
def preData(path, path1):
    f = open(path, "r")  # 源文件
    write = open(path1, "w")
    sequence = []
    for s in f.readlines():
        if len(s) == 1:
            continue
        else:
            tem = s.split()
            if tem[0] == "6881":

                line = "2068" + " " + tem[1] + " " + tem[2] + " " + tem[3] + " " + tem[-2] + " " + tem[-1] + "\n"
                write.write(line)

    f.close()


if __name__ == "__main__":
    path1 = "E:\\datacollect\\test\\data\\"
    path2 = ""

    for p in os.listdir(path1):
        p1 = path1 + p
        p2 = path1 + p + "tem.txt"
        preData(p1, p2)
