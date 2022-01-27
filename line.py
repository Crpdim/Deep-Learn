import random
from numpy import indices
import torch
from d2l import torch as d2l
def syn(w,b,num_example):   #构造数据集
    x = torch.normal(0,1,(num_example,len(w))) #均值为0，方差为1，n个样本，w长度个列
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape) #加入均值为0，方差为0.1的随机噪声
    return x,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4]) #真实w
true_b = 4.2                    #真实b
features,labels = syn(true_w,true_b,1000) #生成一千条训练数据，得到featyres，labels
# print(features[0],"/n",labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:,(1)].detach().numpy(),labels.detach().numpy(),1) #从计算图中detach出来，转numpy
# d2l.plt.show()

def data_iter(batch_size,feature,labels):   #批量读取样本
    num_example = len(feature)
    indices = list(range(num_example)) 
    random.shuffle(indices)                 #随机读取
    for i in range(0,num_example,batch_size):
        batch_indices = torch.tensor(indices[i:
            min(i + batch_size, num_example)])
        yield feature[batch_indices],labels[batch_indices]

batchSize = 10
# for X, y in data_iter(batchSize,features,labels):
#     print(X,"\n",y)
#     break

#初始化模型参数
w = torch.normal(0,0.01,size = (2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True) #需要对w和b进行更新，requires_frad = True
#定义模型
#线性回归方程
def linreg(X,w,b):
    return torch.matmul(X,w) + b
#定义损失函数
def squared_loss(y_hat,y):
    #y_hat ： 实际值
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
 
#定义优化算法
#小批量梯度下降
#param：参数
#lr：学习率
#batch： 步长
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # print(param.grad)
            param.grad.zero_()
# batchSize = 10
lr = 0.03           #学习率
num_epochs = 3      #训练次数
net = linreg        #回归方程
loss = squared_loss #平房损失函数

for epoch in range(num_epochs): 
    for X,y in data_iter(batchSize,features,labels):
        l = loss(net(X,w,b),y) 
        l.sum().backward() #回归方程求梯度
        sgd([w,b],lr,batchSize) 
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f"epoch{epoch + 1 }, loss {float(train_l.mean()):f}")

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')