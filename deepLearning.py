from __future__ import print_function, division 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import sampler 
import torchvision.datasets as dset 
import torchvision.transforms as T 

from sklearn import metrics

torch.set_default_tensor_type(torch.cuda.FloatTensor)

import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os 
import datetime 
import numpy as np 
import math 

## 文件数据格式: x y z time;表示一个坐标点的三个坐标分量和采集时间
def pre(source, distance, diantancel):
    f = open(source, "r") #源文件
    fwrit = open(distance, "a")
    for s in f.readlines():
        if len(s) == 1:
            fwrit.write(s) 
        else:
            s = s[:-1]
            tem = s.split() 
            re = tem[1].split(':')[1] + "\t" + tem[2].split(':')[1] + "\t" + tem[3].split(':')[1] + "\t" + tem[-2] + "\t" + tem[-1] + "\n"
            if tem[0] == "2068":
                fwrit.write(re)
        
    f.close()
    fwrit.close()

def file_name(file_dir, target, target1):
    path = [file_dir + '\\' + x for x in os.listdir(file_dir)]
    for p in path:
        if not os.path.isdir(p):
            pre(p, target, target1)
    
def split_data(splot):
    state = "D:\\fig\\data\\pre2068Static.txt"
    unrealize = "D:\\fig\\data\\pre2068Unrealize.txt"
    Sactive = "D:\\fig\\data\\pre2068Little.txt"
    Mactive = "D:\\fig\\data\\pre2068LargeMove.txt"
    files = [state, unrealize, Sactive, Mactive]
    mask = [0., 1., 2., 3.]
    splotre = []
    lable = [] 
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
                if flag == False :
                    #本数据序列第一点的采集时间
                    start = 0
                    now = 0
                    flag = True 
                # 当前点的采集时间
                now = now + 1
                subt = now - start 
                obj.append(np.asarray(seq[:3],dtype='float64'))
            if subt > splot:
                splotre.append(np.asarray(obj))
                lable.append(mk)
                obj.clear() 
                flag = False
    splotre = np.asarray(splotre)
    lable = np.asarray(lable)
    return splotre, lable

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

splotre, lable = split_data(39)

NUM_TRAIN = 25000

index = list(range(splotre.shape[0]))
# 索引洗牌 等价于随机抽样
np.random.shuffle(index)
splotre = splotre[index]
lable = lable[index]
train_splot = splotre[:NUM_TRAIN-1000]
train_lable = lable[:NUM_TRAIN-1000]
val_splot = splotre[NUM_TRAIN-1000:NUM_TRAIN]
val_lable = lable[NUM_TRAIN-1000:NUM_TRAIN]
test_splot = splotre[NUM_TRAIN:]
test_lable = lable[NUM_TRAIN:]


class ToTensor(object):
    def __call__(self, splot):
        return torch.from_numpy(splot)
    
trans = T.Compose([
    ToTensor()
])

class LocationDataset(Dataset):
    def __init__(self, splotre, lable, transform = trans):
        self.splotre = np.transpose(splotre, (0, 2, 1))
        self.lable = lable 
        self.transform = transform 
    
    def __len__(self):
        return len(self.splotre)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        splot = self.splotre[idx] 
        lable = self.lable[idx]
        tensor = trans(splot)
        return tensor, lable
train_dataset = LocationDataset(train_splot, train_lable)
loader_train = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataset = LocationDataset(val_splot, val_lable)
loader_val = DataLoader(val_dataset, batch_size=4, shuffle=True)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

import torch.nn.functional as F  # useful stateless functions
def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

def check_accuracy_part34(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

channel_1 = 32 
channel_2 = 16
learning_rate = 1e-4
model = None 
optimizer = None 

model = nn.Sequential(
    nn.Conv1d(3, 32, 3, stride = 1),
    nn.ReLU(),
    nn.Conv1d(32, 64, 2, stride = 2),
    nn.ReLU(),
    nn.Conv1d(64, 128, 3, stride = 2),
    nn.ReLU(),
    Flatten(),
    nn.Linear(512, 4)
)
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, nesterov = True)
train_part34(model, optimizer,5)

test_dataset = LocationDataset(test_splot, test_lable)
loader_test = DataLoader(test_dataset)

def test(loader,model):
    num_correct = 0 
    num_samples = 0 
    res = [] 
    for x, y in loader:
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        res.append(preds.item())
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



res = test(loader_test,model)

def gradeOfClassifier(test_lable, pre):
    P = metrics.precision_score(test_lable, pre, average='macro')
    R = metrics.recall_score(test_lable, pre, average='macro')
    # F1分数
    F1 = metrics.f1_score(test_lable, pre, average='weighted')
    # 混淆矩阵
    M = metrics.confusion_matrix(test_lable, pre, labels=[1.0, 2.0, 3.0, 4.0])
    print("查准率" + str(P))
    print("召回率" + str(R))
    print("F1分数" + str(F1))
    print("混淆矩阵")
    print(M)
    print()
    return P
gradeOfClassifier(test_lable, res)


    