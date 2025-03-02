import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from Net import ResNet50

# 定义类别标签（CIFAR-10 的类别）
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 加载模型
def load_model(model_path, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载预训练的 ResNet-50 模型
    model = ResNet50().to(device)#引入自己的模型
    #model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改最后一层
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置为评估模式
    return model

# 图片预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图片尺寸为 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 的均值和方差
    ])
    image = Image.open(image_path).convert('RGB')  # 打开图片并转换为 RGB 格式
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    return image

# 预测图片类别
def predict_image(image_path, model, device):
    # 预处理图片
    image = preprocess_image(image_path).to(device)

    # 预测
    with torch.no_grad():
        model.eval()
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class

# 主函数
def main():
    # 检查是否有 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_path = r'PaperRecurrence\ResNet\cifar10best_model.pth'  # 模型文件路径
    model = load_model(model_path).to(device)

    # 测试图片路径
    image_path = r'PaperRecurrence\ResNet\cat.1535.jpg'  # 替换为你的测试图片路径

    # 预测图片类别
    predicted_class = predict_image(image_path, model, device)
    print(f'Predicted class: {predicted_class}')

if __name__ == '__main__':
    main()