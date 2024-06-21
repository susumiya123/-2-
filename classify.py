import torch     #导入依赖的各种库
from torchvision import models, transforms
from PIL import Image
from pytorch_lightning.core.saving import load_hparams_from_yaml
from torch import nn
import numpy as np
class ViolenceClass:
    def __init__(self, model_path, device):  #模型初始化函数，接受参数为预训练的模型权重和使用设备
        # 设置设备
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建一个ResNet18模型实例，不加载预训练权重，并设置输出类别数为2
        #self.model = models.resnet18(pretrained=False)
        self.model = models.resnet18(pretrained=False, num_classes=2)
        # 加载预训练模型权重
        self.model = self.model.to(self.device) # 指定模型使用的设备
        self.model.eval()         # 设置模型为评估模式

        # 加载模型的checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict']   # 提取状态字典
        new_state_dict = {}
        for key in state_dict.keys():
            if 'model.' in key:
                new_state_dict[key.replace('model.', '')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        self.model.load_state_dict(new_state_dict)  # 把重新组装的状态字典加载到模型中
        
        # 定义图像变换操作，包括调整图像大小和转换为Tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def classify(self, imgs: torch.Tensor) -> list:  #图形分类函数，接收 Tensor 格式的图像数据，对图像进行分类
        
        self.model.eval() # 确保模型在评估模式

        
        imgs = imgs.to(self.device) # 将图像数据移动到指定设备
        
        with torch.no_grad():
            logits = self.model(imgs) # 执行模型推理，获取未归一化的预测值
        
        # 获取预测类别
        _, preds = torch.max(logits, 1)

        
        preds = preds.cpu().tolist()  ## 将预测结果转换为 要求的Python 列表
        # 输出所有图像的预测结果
        predictions_array = np.array(preds)
        print(predictions_array)
        return preds