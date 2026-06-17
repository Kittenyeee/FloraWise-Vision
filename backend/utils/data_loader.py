"""数据加载器模块"""
import os
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class FlowerDataset(Dataset):
    """花卉数据集类
    
    支持从本地目录加载数据
    目录结构:
        data_dir/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── class2/
        │   └── ...
        └── ...
    """
    
    def __init__(self, data_dir: str, transform=None, is_train: bool = False):
        """初始化数据集
        
        Args:
            data_dir: 数据集目录
            transform: 图像转换管道
            is_train: 是否为训练集
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """加载数据集"""
        if not self.data_dir.exists():
            raise ValueError(f"数据集目录不存在: {self.data_dir}")
        
        # 获取类别列表
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        if not class_dirs:
            raise ValueError(f"未找到任何类别目录: {self.data_dir}")
        
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 加载图像路径和标签
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_idx)
        
        if not self.images:
            raise ValueError(f"未找到任何图像文件: {self.data_dir}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (图像张量, 标签)
        """
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"加载图像失败 {self.images[idx]}: {e}")
            # 返回随机张量作为备选
            return torch.randn(3, 224, 224), self.labels[idx]
    
    def get_class_info(self) -> Dict:
        """获取数据集信息"""
        class_counts = {}
        for class_name in self.class_names:
            class_counts[class_name] = sum(1 for label in self.labels 
                                          if label == self.class_to_idx[class_name])
        
        return {
            'total_samples': len(self.images),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': class_counts
        }


class FlowerDataLoader:
    """数据加载器管理类"""
    
    # 标准化参数
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 224
    
    def __init__(self, data_root: str, batch_size: int = 32, 
                 num_workers: int = 0, input_size: int = 224):
        """初始化数据加载器
        
        Args:
            data_root: 数据集根目录
            batch_size: 批次大小
            num_workers: 数据加载线程数
            input_size: 输入图像尺寸
        """
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        
        # 定义转换管道
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        train_dir = self.data_root / 'train'
        dataset = FlowerDataset(str(train_dir), transform=self.train_transform, is_train=True)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        ), dataset
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        val_dir = self.data_root / 'val'
        dataset = FlowerDataset(str(val_dir), transform=self.val_transform, is_train=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ), dataset
