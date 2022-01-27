from importlib.resources import is_resource
import random
import torch
from torch.utils import data
import numpy as np
from d2l import torch as d2l
from line import data_iter

true_w = torch.tensor([2,-3.4])
true_b = 4.2
feature,labels = d2l.synthetic_data(true_w,true_b,1000) #构造数据集
def load_array(data_arrays, batch_size, is_train = True):
    """构造pytorch迭代器"""
    dataset = data.TensorDataset(*data_arrays) #读取数据集
    return data.DataLoader(dataset,batch_size, shuffle = is_train) 
    #data.DataLoader()每次挑选batch_size个样本，suffle选择是否随机打乱

batch_size = 10
data_iter = load_array((feature, labels),batch_size)
#next() 返回迭代器的下一个项目。
print(next(iter(data_iter)))

#定义模型
from torch import nn
#指定输入输出层
net = nn.Sequential(nn.Linear(2,1)) 
#神经网络第一层初始化模型参数
net[0].weight.data.normal_(0,0.01) #w 使用正态分布替换w的值
net[0].bias.data.fill_(0) #b
#定义均方误差
loss = nn.MSELoss() 
#实例化SGD
#第一个参数包括net所有参数（w，b），第二个参数指定学习率
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)
num_epoch = 3

for epoch in range(num_epoch):
    for X,y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(feature),labels)
    print(f"epoch{epoch + 1},loss {l:f}")



