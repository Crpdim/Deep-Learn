import torch
from torch import nn
from d2l import torch as d2l
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
batch_size = 256
##加载数据集函数在softmax从零实现中已经实现过,这里直接调用
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#初始化模型参数,将每个图像视为具有784个输入特征 和10个类的简单分类数据集
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# 定义各个隐藏层权重和偏差
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]
#ReLU激活函数实现
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

#定义网络模型
#使用reshape将每个二维图像转换为一个长度为num_inputs的向量。
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
#调用内置损失函数loss
loss = nn.CrossEntropyLoss()

#训练
#多层感知机的训练过程与softmax回归的训练过程完全相同,可以直接使用softmax的训练函数
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
#模型评估
d2l.predict_ch3(net, test_iter)
d2l.plt.show()

#简洁实现
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)