#Res-Net50复现
#猫狗数据集不大好，Res-Net50的参数大约有2.5亿，训练集只有2000张图片，测试集只有1000张图片，无法提供足够的数据供学习
#未知数的量可能大于等式量，无法求解
#CPU跑的太慢了······
#所以改用CIFAR-10数据集
#CIFAR-10数据集是一个用于识别普适物体的小型数据集，共有10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import os
import PIL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import pathlib

def main():
  # 检查是否有 GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  t0=time.time()
  #查看数据信息
  data_dir_train=r'data\Cat_Dog\train'
  data_dir_test=r'data\Cat_Dog\test'
  data_dir_train=pathlib.Path(data_dir_train)#转换为pathlib格式
  data_dir_test=pathlib.Path(data_dir_test)
  print(data_dir_train)
  print(data_dir_test)
  t1=time.time()
  print('Time used in load data & Packages:',t1-t0)

  #类别数量(为文件夹的数量，不同数据存放方式会不同)
  classnames=os.listdir(data_dir_train)
  num_classes=len(classnames)
  print('Class name:',classnames)
  print('Number of class:',num_classes)

  #展示部分数据
  #展示部分数据
  import matplotlib.image as mpimg
  import random

  #展示数据
  def show_data(data_dir):
      plt.figure(figsize=(20,10))
      for i in range(10):
          plt.subplot(2,5,i+1)
          class_name=random.choice(os.listdir(data_dir))
          img_name=random.choice(os.listdir(data_dir/class_name))
          img_path=data_dir/class_name/img_name
          img=mpimg.imread(img_path)
          plt.imshow(img)
          plt.title(class_name)
          plt.axis('off')
      plt.show()
  #show_data(data_dir_train)

  #数据预处理及载入
  from torchvision import datasets,transforms
  img_height,img_width=256,256
  batch_size=32
  transform_train=transforms.Compose([
      transforms.Resize((img_height,img_width)),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#normalize后面的参数是均值和方差，三个值分别对应三个通道，参数来自cifar10数据集
  ])#normalize后面的参数是均值和方差，三个值分别对应三个通道，参数来自cifar10数据集

  transform_test = transforms.Compose([
      transforms.Resize((img_height,img_width)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  '''
  #数据划分（在此数据上已经划分完毕，无需此步操作）
  train_size=int(0.7*len(train_data))
  test_size=len(train_data)-train_size
  train_data,test_data=torch.utils.data.random_split(train_data,[train_size,test_size])
  train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
  '''

  #载入数据
  '''
  train_data=datasets.ImageFolder(data_dir_train,data_transforms)
  #datasets.ImageFolder会自动将类别名转换为数字标签（从0开始）。例如，cat对应0，dog对应1。
  train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)

  #测试集只有一个文件夹，不需要再分
  #载入只有一个文件夹的测试集
  test_data=datasets.ImageFolder(data_dir_test,data_transforms)
  test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
  '''
  # 加载 CIFAR-10 数据集
  train_data = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
  test_data = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)#num_workers是线程数
  test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

  #输出处理后的图像尺寸
  dataiter=iter(test_loader)
  #dataiter=iter(train_loader)
  images,labels=next(dataiter)
  print('Image Shape[BatchSize,Channel,Height,Width]:',images.size())#参数分别是batch_size,channel,height,width
  print('Labels:',labels.size(),labels)#参数是batch_size，32表示一个batch有32个样本，labels是目前一个batch的标签


  t2=time.time()
  print('Time used in data processing:',t2-t1)

  # ResNet-50 网络结构

  '''
  输入层：
  - 输入：3通道的RGB图像（例如224x224x3）。
  '''

  '''
  1. 初始卷积层：
  - Zero Padding：在图像周围填充3个0（padding=3）。
  - Conv1：7x7卷积，64个过滤器，步幅为2（stride=2），填充为3（padding=3）。
    - 作用：提取低级特征（如边缘、纹理）。
  - BN：批量归一化，加速训练并稳定梯度。
  - ReLU：激活函数，引入非线性。
  - MaxPool：3x3最大池化，步幅为2（stride=2），填充为1（padding=1）。
    - 作用：进一步降低特征图尺寸，增加感受野。
  '''

  '''
  2. 第一阶段（Conv Block *1 + Identity Block *2）：
  - Conv Block *1：
    - Long Path：
      - Conv 1x1：64个过滤器，步幅为1（stride=1），用于降维。
      - BN：批量归一化。
      - ReLU：激活函数。
      - Conv 3x3：64个过滤器，步幅为1（stride=1），填充为1（padding=1），用于提取特征。
      - BN：批量归一化。
      - ReLU：激活函数。
      - Conv 1x1：256个过滤器，步幅为1（stride=1），用于升维。
      - BN：批量归一化。
    - Shortcut Path：
      - Conv 1x1：256个过滤器，步幅为1（stride=1），用于调整输入维度。
      - BN：批量归一化。
    - Add：将Long Path和Shortcut Path的输出相加。
    - ReLU：激活函数。
  - Identity Block *2：
    - Long Path：
      - Conv 1x1：64个过滤器，步幅为1（stride=1），用于降维。
      - BN：批量归一化。
      - ReLU：激活函数。
      - Conv 3x3：64个过滤器，步幅为1（stride=1），填充为1（padding=1），用于提取特征。
      - BN：批量归一化。
      - ReLU：激活函数。
      - Conv 1x1：256个过滤器，步幅为1（stride=1），用于升维。
      - BN：批量归一化。
    - Shortcut Path：直接使用输入（x）。
    - Add：将Long Path和Shortcut Path的输出相加。
    - ReLU：激活函数。
  '''

  '''
  3. 第二阶段（Conv Block *1 + Identity Block *3）：
  - Conv Block *1：
    - 结构与第一阶段的Conv Block类似，但输出通道数增加为512。
  - Identity Block *3：
    - 结构与第一阶段的Identity Block类似，但输出通道数增加为512。
  '''

  '''
  4. 第三阶段（Conv Block *1 + Identity Block *5）：
  - Conv Block *1：
    - 结构与第一阶段的Conv Block类似，但输出通道数增加为1024。
  - Identity Block *5：
    - 结构与第一阶段的Identity Block类似，但输出通道数增加为1024。
  '''

  '''
  5. 第四阶段（Conv Block *1 + Identity Block *2）：
  - Conv Block *1：
    - 结构与第一阶段的Conv Block类似，但输出通道数增加为2048。
  - Identity Block *2：
    - 结构与第一阶段的Identity Block类似，但输出通道数增加为2048。
  '''

  '''
  6. 全局平均池化层：
  - Average Pooling：7x7池化，步幅为1（stride=1），填充为0（padding=0）。
    - 作用：将特征图尺寸降为1x1，减少参数数量。
  '''

  '''
  7. 全连接层：
  - Flatten：将多维数据展平为一维。
  - Dense：全连接层，输出类别概率（使用Softmax激活函数）。
    - 作用：根据提取的特征进行分类。
  '''
  #构建模型

  class ResNet50(nn.Module):
      def __init__(self, num_classes=10):
          super(ResNet50, self).__init__()

          # 初始卷积层
          #nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
          self.conv1 = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=7, stride=7, padding=3, bias=False),
              nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
              #eps：分母中添加的值，避免分母为0,momentum：动量，affine：是否进行仿射变换，track_running_stats：是否追踪运行时的统计信息
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
              #dilation：控制卷积核元素之间的间距，ceil_mode：是否向上取整
          )

          # 定义 Conv Block
          def conv_block(in_channels, out_channels, stride=1):
              return nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(out_channels * 4,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )

          # 定义 Identity Block
          def identity_block(in_channels, out_channels):
              return nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False),
                  nn.BatchNorm2d(out_channels * 4,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )

          # 第一阶段
          self.stage1 = nn.Sequential(
              conv_block(64, 64, stride=1),  # Conv Block
              identity_block(256, 64),       # Identity Block
              identity_block(256, 64)        # Identity Block
          )

          # 第二阶段
          self.stage2 = nn.Sequential(
              conv_block(256, 128, stride=2),  # Conv Block
              identity_block(512, 128),        # Identity Block
              identity_block(512, 128),        # Identity Block
              identity_block(512, 128)         # Identity Block
          )

          # 第三阶段
          self.stage3 = nn.Sequential(
              conv_block(512, 256, stride=2),  # Conv Block
              identity_block(1024, 256),       # Identity Block
              identity_block(1024, 256),       # Identity Block
              identity_block(1024, 256),       # Identity Block
              identity_block(1024, 256),       # Identity Block
              identity_block(1024, 256)        # Identity Block
          )

          # 第四阶段
          self.stage4 = nn.Sequential(
              conv_block(1024, 512, stride=2),  # Conv Block
              identity_block(2048, 512),        # Identity Block
              identity_block(2048, 512)         # Identity Block
          )

          # 全局平均池化层
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出尺寸为 1x1
          #kernel_size：池化窗口大小，stride：步幅，padding：填充

          # 全连接层
          self.fc = nn.Linear(2048, num_classes, bias=True)   

      def forward(self, x):
          # 初始卷积层
          x = self.conv1(x)

          # 第一阶段
          x = self.stage1(x)

          # 第二阶段
          x = self.stage2(x)

          # 第三阶段
          x = self.stage3(x)

          # 第四阶段
          x = self.stage4(x)

          # 全局平均池化层
          x = self.avgpool(x)

          # 全连接层
          x = torch.flatten(x, 1)
          x = self.fc(x)

          return x

  #模型实例化

  model=ResNet50().to(device)
  t3=time.time()
  print('Time used in model building:',t3-t2)

  #定义损失函数和优化器
  #损失函数
  loss_fn=nn.CrossEntropyLoss()
  #优化器
  optimizer=optim.Adam(model.parameters(),lr=0.01)
  # 定义学习率调度器
  scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

  #构建训练
  def train(data_loader, model, loss_fn, optimizer, num_epochs):
      size = len(data_loader.dataset)  # 数据集大小
      batch_num = len(data_loader)  # batch数量
      model.train()  # 将模型设置为训练模式

      # 用于记录每个epoch的损失和准确率
      train_loss_list = []
      train_acc_list = []

      for epoch in range(num_epochs):
          train_acc = 0.0
          train_loss = 0.0
          t_train_start = time.time()
          for images, labels in data_loader:
              # 将数据移动到设备（如GPU）
              images = images.to(device)
              labels = labels.to(device)

              # 前向传播
              outputs = model(images)
              loss = loss_fn(outputs, labels)

              # 反向传播和优化
              optimizer.zero_grad()  # 梯度清零
              loss.backward()  # 反向传播
              optimizer.step()  # 更新参数

              # 计算准确率和损失
              train_acc += (outputs.argmax(1) == labels).float().sum().item()  # 累加正确预测的数量
              train_loss += loss.item()  # 累加损失

              #scheduler.step(loss)  # 更新学习率
              
          t_train_end = time.time()
          print(f'Total training time: {t_train_end - t_train_start:.2f} seconds')

          # 计算每个epoch的平均损失和准确率
          train_acc /= size
          train_loss /= batch_num  # 平均损失 = 总损失 / batch数量

          # 记录结果
          train_loss_list.append(train_loss)
          train_acc_list.append(train_acc)


      return train_loss_list, train_acc_list

  #构建测试
  def test(data_loader, model, loss_fn):
      size = len(data_loader.dataset)  # 数据集大小
      num_batches = len(data_loader)  # batch数量
      model.eval()  # 将模型设置为评估模式
      test_loss = 0.0
      test_acc = 0.0

      with torch.no_grad():  # 关闭梯度计算
          for images, labels in data_loader:
              # 将数据移动到设备（如GPU）
              images = images.to(device)
              labels = labels.to(device)

              # 前向传播
              outputs = model(images)
              test_loss += loss_fn(outputs, labels).item()  # 累加损失
              test_acc += (outputs.argmax(1) == labels).float().sum().item()  # 累加正确预测的数量

      # 计算平均损失和准确率
      test_loss /= num_batches  # 平均损失 = 总损失 / batch数量
      test_acc /= size  # 准确率 = 正确预测的数量 / 数据集大小




      # 打印测试结果
      print(f'Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc * 100:.2f}%')

      return test_loss, test_acc

  #训练
  num_epochs = 40
  best_test_acc = 0.0
  for epoch in range(num_epochs):
      train_loss_list, train_acc_list = train(train_loader, model, loss_fn, optimizer, num_epochs=1)
      
      # 打印每个epoch的结果
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss_list[-1]:.5f}, Accuracy: {train_acc_list[-1]* 100:.2f}%')
      #测试
      test_loss, test_acc = test(test_loader, model, loss_fn)
      scheduler.step(test_acc)  # 更新学习率

      if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved best model with test accuracy: {best_test_acc*100:.2f}%')
        print(f'Saved model for epoch {epoch+1}')
      # 每个 epoch 结束后保存模型
      #torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
      #print(f'Saved model for epoch {epoch+1}')

if __name__ == '__main__':
    main()
else:
    # 导出 ResNet50 类
    __all__ = ['ResNet50']



'''输出
Total training time: 85.62 seconds
Epoch [39/40], Loss: 0.42612, Accuracy: 84.85%
Test Loss: 0.77307, Test Accuracy: 74.75%
Saved best model with test accuracy: 74.75%
Saved model for epoch 39
Total training time: 84.38 seconds
Epoch [40/40], Loss: 0.42780, Accuracy: 84.88%
Test Loss: 0.78341, Test Accuracy: 74.51%


'''






