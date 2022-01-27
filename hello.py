import torch

x = torch.arange(4.0) 
#x = tensor([0., 1., 2., 3.])
x.requires_grad_(True) #存储x的梯度，x.grad访问梯度
print (x.grad)
y = 2 * torch.dot(x,x) #x和x的点内积，乘2
y.backward() #调用反向传递函数 计算y关于x每个分量的梯度
print(x.grad) 
print(x.grad == 4*x)
#默认情况下，pytorch会把梯度累计,再下一次使用时要清空梯度
x.grad.zero_() #把所有梯度清零
y = torch.dot(x,x)
y.backward()
print(x.grad)
#将某些计算移动到计算图外
x.grad.zero_()
y = x*x
u = y.detach() #将u视为常数
z = u*x
z.sum().backward()
print(x.grad)