"""图像预处理模块"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union
import torchvision.transforms as transforms


class ImagePreprocessor:
    """图像预处理类"""
    
    # 标准化参数（ImageNet）
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = 224
    
    def __init__(self, input_size: int = 224):
        """初始化预处理器
        
        Args:
            input_size: 输入图像尺寸，默认224
        """
        self.input_size = input_size
        
        # 定义转换管道
        self.transform_train = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
        
        self.transform_val = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image], 
                        is_train: bool = False) -> np.ndarray:
        """预处理单张图像
        
        Args:
            image: 图像路径、numpy数组或PIL Image
            is_train: 是否使用训练时的数据增强
            
        Returns:
            预处理后的张量
        """
        # 读取图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.convert('RGB')
        
        # 应用转换
        transform = self.transform_train if is_train else self.transform_val
        return transform(img)
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像数组 (H, W, C)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小
        
        Args:
            image: 输入图像
            size: 目标尺寸 (H, W)
            
        Returns:
            调整后的图像
        """
        return cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean=None, std=None) -> np.ndarray:
        """标准化图像
        
        Args:
            image: 输入图像 (H, W, C) 或 (C, H, W)
            mean: 均值
            std: 标准差
            
        Returns:
            标准化后的图像
        """
        if mean is None:
            mean = ImagePreprocessor.MEAN
        if std is None:
            std = ImagePreprocessor.STD
        
        # 转换为 (C, H, W) 格式
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)
        
        # 标准化
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[i] = (image[i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def augment_image(image: np.ndarray, random_seed: int = None) -> np.ndarray:
        """数据增强
        
        Args:
            image: 输入图像
            random_seed: 随机种子
            
        Returns:
            增强后的图像
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 随机翻转
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # 随机旋转
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
        
        # 随机亮度调整
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        
        return image
