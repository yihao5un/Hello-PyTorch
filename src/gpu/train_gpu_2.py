import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义训练的设备
# device = torch.device("cpu") # 利用CPU
device = torch.device("cuda:0")  # 利用GPU 还可以指定具体的GPU

# 准备训练数据集
train_data = torchvision.datasets.CIFAR10(root="../../data", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10(root="../../data", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
# 准备训练和测试数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度 {}", format(train_data_size))
print("测试数据集的长度 {}", format(test_data_size))

# 利用DataLoader进行加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=64, num_workers=0)


# 创建网络模型

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


yihao = Yihao()
# 利用device训练
yihao = yihao.to(device)

# 将网络模型转移到 CUDA 利用 GPU 训练
# if torch.cuda.is_available():
#     yihao = yihao.cuda()

# 损失函数 交叉熵
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 将网络模型转移到 CUDA 利用 GPU 训练
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-3  # (1 * (10) ^ -2)
optimizer = torch.optim.SGD(yihao.parameters(), lr=learning_rate, momentum=0, weight_decay=0)

# 设置训练网络的一些参数

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10


def train_optimizer():
    # ==== 优化器优化模型 === #
    # 优化器的梯度清零
    optimizer.zero_grad()
    # 损失反向传播
    loss.backward()
    # 根据优化器的规则（如学习率、动量等）和计算出的梯度，更新模型的参数
    optimizer.step()


# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("--------- 第 {} 轮训练开始 ---------".format(i + 1))
    # 训练步骤开始 !!!
    yihao.train()
    for data in train_dataloader:
        imgs, targets = data
        # 将网络模型转移到 CUDA 利用 GPU 训练
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 将训练数据放到网络中
        outputs = yihao(imgs)
        # 将预测输出outputs和真实targets放到损失函数当中
        loss = loss_fn(outputs, targets)
        # 优化器
        train_optimizer()
        # 训练次数加一
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数 {}, Loss: {}".format(total_train_step, loss.item()))
            # tensorboard
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    yihao.eval()
    total_test_loss = 0
    # 整体正确个数
    total_accuracy = 0
    # 将梯度设置去掉 torch.no_grad()
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 将网络模型转移到 CUDA 利用 GPU 训练
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = yihao(imgs)
            loss = loss_fn(outputs, targets)
            # 把每一次的loss加到整体的loss上
            total_test_loss = total_test_loss + loss.item()
            # 计算当前批次的正确预测数量
            accuracy = (outputs.argmax(1) == targets).sum()
            # 累加正确预测数量
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    # tensorboard
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step = total_test_step + 1

    # 保存模型

    # torch.save(yihao, "yihao_{}.pth".format(i)) # 方式一
    torch.save(yihao.state_dict(), "yihao_{}.pth".format(i))  # 方式二
    print("模型已保存")

writer.close()
