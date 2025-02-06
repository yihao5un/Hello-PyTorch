# 搭建神经网络
import torch
from torch import nn


class Yihao(nn.Module):
    def __init__(self):
        super(Yihao, self).__init__()
        #  借助 CIFAR10 模型结构理解卷积神经网络及Sequential的使用 https://blog.csdn.net/m0_48241022/article/details/132634215
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    yihao = Yihao()
    input = torch.ones(64, 3, 32, 32)
    output = yihao(input)
    print(output.shape)