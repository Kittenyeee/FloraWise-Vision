"""推理引擎模块"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np
from .preprocessing import ImagePreprocessor


class InferenceEngine:
    """推理引擎类
    
    负责加载模型、执行推理、处理结果
    """
    
    def __init__(self, model, class_names: List[str], device: str = 'cpu'):
        """初始化推理引擎
        
        Args:
            model: PyTorch模型
            class_names: 类别名称列表
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.preprocessor = ImagePreprocessor()
    
    @torch.no_grad()
    def predict(self, image_input, confidence_threshold: float = 0.0) -> Dict:
        """执行单张图像推理
        
        Args:
            image_input: 图像路径、numpy数组或PIL Image
            confidence_threshold: 置信度阈值
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 预处理图像
            image_tensor = self.preprocessor.preprocess_image(image_input, is_train=False)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 推理
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            
            # 获取预测结果
            top_class_idx = np.argmax(probabilities)
            top_confidence = float(probabilities[top_class_idx])
            
            # 检查置信度
            if top_confidence < confidence_threshold:
                return {
                    'success': False,
                    'error': f'置信度 ({top_confidence:.2%}) 低于阈值 ({confidence_threshold:.2%})'
                }
            
            # 构建结果
            predictions = []
            for idx, conf in enumerate(probabilities):
                predictions.append({
                    'class': self.class_names[idx],
                    'confidence': float(conf),
                    'rank': idx + 1
                })
            
            # 按置信度排序
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'success': True,
                'top_class': self.class_names[top_class_idx],
                'confidence': top_confidence,
                'all_predictions': predictions,
                'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probabilities)}
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'推理失败: {str(e)}'
            }
    
    @torch.no_grad()
    def batch_predict(self, image_list: List, confidence_threshold: float = 0.0) -> List[Dict]:
        """批量推理
        
        Args:
            image_list: 图像列表
            confidence_threshold: 置信度阈值
            
        Returns:
            推理结果列表
        """
        results = []
        for image in image_list:
            result = self.predict(image, confidence_threshold)
            results.append(result)
        return results
    
    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'device': self.device,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
