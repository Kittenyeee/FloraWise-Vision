"""花卉分类模型定义"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class FlowerClassifier(nn.Module):
    """花卉分类模型（基于MobileNetV2的迁移学习）
    
    Args:
        num_classes: 分类数量，默认5（菊花、蒲公英、玫瑰、向日葵、郁金香）
        pretrained: 是否使用预训练权重，默认True
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(FlowerClassifier, self).__init__()
        
        # 加载预训练的MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # 获取特征提取器的输出维度
        in_features = self.backbone.classifier[1].in_features
        
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, 224, 224)
            
        Returns:
            输出张量，形状为 (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self) -> None:
        """冻结骨干网络的权重，只训练分类头"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """解冻骨干网络，允许训练所有层"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> dict:
        """获取模型信息
        
        Returns:
            包含模型参数统计的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MobileNetV2',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }


class SimpleResNet(nn.Module):
    """简化版ResNet18分类器（备选方案）
    
    Args:
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(SimpleResNet, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
