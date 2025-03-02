# ResNet50-
ResNet50论文复现，模型还有待进一步学习，欢迎大家喂更多的数据来进一步训练
---

# ResNet-50 实现与训练

本项目实现了经典的 ResNet-50 模型，并在 CIFAR-10 数据集上进行了训练和测试。代码支持 GPU 加速训练，并提供了模型保存和加载功能，方便后续使用。



## 环境要求

- Python 3.8+
- PyTorch 1.10+
- torchvision
- PIL (Pillow)
- matplotlib (可选，用于可视化)

### 安装依赖

---

## 数据集

本项目使用 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 数据集，包含 10 个类别的 60,000 张 32x32 彩色图片。

- 训练集：50,000 张图片
- 测试集：10,000 张图片

数据集会自动下载并保存在 `data/cifar10` 目录下。

---

## 使用方法

### 1. 训练模型

运行以下命令开始训练 ResNet-50 模型：

```bash
python Res50.py
```

- 训练过程中会保存最佳模型到 `models/best_model.pth`。
- 训练日志会实时打印，包括每个 epoch 的损失和准确率。

### 2. 测试模型

运行以下命令测试模型在测试集上的性能：

```bash
python Res50.py
```

- 测试结果会打印测试集的损失和准确率。

### 3. 使用模型进行图片分类

运行以下命令对单张图片进行分类：

```bash
python ModelTest.py --image_path test_image.jpg
```

- 将 `test_image.jpg` 替换为你要分类的图片路径。
- 脚本会输出预测的类别名称。

---

## 模型性能

在 CIFAR-10 数据集上，ResNet-50 模型的性能如下：

- **训练集准确率**：85%
- **测试集准确率**：75%

---

## 代码说明

### `Res50.py`

- 定义了 ResNet-50 模型结构。
- 包含数据加载、模型训练和测试的逻辑。
- 支持 GPU 加速训练。
- 训练过程中会保存最佳模型。

### `ModelTest.py`

- 加载训练好的模型权重。
- 对输入的图片进行预处理和分类。
- 输出预测的类别名称。

---

## 自定义数据集

如果你想在其他数据集上训练模型，可以按照以下步骤操作：

1. 将数据集放置在 `data/` 目录下。
2. 修改 `Res50.py` 中的数据加载部分，适配你的数据集格式。
3. 调整模型的输出类别数（`num_classes`）。

---

## 贡献

欢迎提交 Issue 或 Pull Request 改进本项目！

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 参考

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - ResNet 论文
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) - 数据集官网
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - PyTorch 官方文档

---

## 联系方式

如有问题或建议，请联系：dengmingxi2003@163.com

---

## 致谢

感谢 PyTorch 团队和 CIFAR-10 数据集提供者！

---

希望这份 `README.md` 文档能帮助你更好地展示和分享你的 ResNet-50 项目！如果有其他需求，欢迎继续提问！
