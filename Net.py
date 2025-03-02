import torch
import torch.nn as nn

# 定义 ResNet-50 模型
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
