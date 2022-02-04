from http.client import ImproperConnectionState
from logging import root
from turtle import down
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()

trans = transforms.ToTensor()
#下载训练数据集
mnist_train = torchvision.datasets.FashionMNIST(root = "/Users/crpdim/Desktop/Deep-Learn/data",
                            train = True,transform=trans,download=True)
#下载测试数据集，用来验证模型好坏
mnist_test = torchvision.datasets.FashionMNIST(root = "/Users/crpdim/Desktop/Deep-Learn/data",train = False,
                                    transform=trans,download=True)

print(mnist_train[0][0].shape)
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
batch_size = 256

def get_dataloader_workers():
    #使用四个进程读取数据,加快读取速度
    return 4
train_iter = data.DataLoader(mnist_train,batch_size,shuffle=True,
            num_workers=get_dataloader_workers())
        
timer = d2l.Timer()
for X,y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


#从零开始softmax
 