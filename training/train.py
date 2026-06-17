"""
模型训练脚本
完整的花卉分类模型训练流程
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import time
import json
from datetime import datetime

from backend.models.flower_model import FlowerClassifier
from backend.utils.data_loader import FlowerDataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlowerModelTrainer:
    """花卉分类模型训练器"""
    
    def __init__(self, config: dict):
        """初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.best_accuracy = 0
        self.train_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = FlowerClassifier(
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )
        self.model.to(self.device)
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 初始化学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # 数据加载器
        self.data_loader = FlowerDataLoader(
            data_root=config['data_path'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            input_size=config['input_size']
        )
        
        # 创建检查点目录
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='训练中', unit='batch')
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'accuracy': correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='验证中', unit='batch')
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': correct / total
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, accuracy: float, is_best: bool = False):
        """保存检查点
        
        Args:
            epoch: 轮数
            accuracy: 验证准确率
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config,
            'train_history': self.train_history
        }
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        logger.info(f"保存检查点: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path} (准确率: {accuracy:.4f})")
    
    def train(self):
        """完整的训练流程"""
        logger.info("=" * 50)
        logger.info("开始训练花卉分类模型")
        logger.info("=" * 50)
        
        # 加载数据
        logger.info(f"加载训练数据: {self.config['data_path']}")
        train_loader, train_dataset = self.data_loader.get_train_loader()
        val_loader, val_dataset = self.data_loader.get_val_loader()
        
        train_info = train_dataset.get_class_info()
        logger.info(f"训练集: {train_info['total_samples']} 张图像")
        logger.info(f"验证集: {val_dataset.get_class_info()['total_samples']} 张图像")
        logger.info(f"类别: {', '.join(train_info['class_names'])}")
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # 日志输出
            logger.info(
                f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.save_checkpoint(epoch, val_acc, is_best=False)
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"训练完成！总耗时: {total_time / 60:.2f} 分钟")
        logger.info(f"最佳验证准确率: {self.best_accuracy:.4f}")
        logger.info("=" * 50)
        
        # 保存训练历史
        self.save_training_history()
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"保存训练历史: {history_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='花卉分类模型训练脚本')
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='./flower_data',
                        help='数据集路径')
    
    # 模型相关参数
    parser.add_argument('--num_classes', type=int, default=5,
                        help='类别数量')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练权重')
    parser.add_argument('--input_size', type=int, default=224,
                        help='输入图像大小')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最小学习率')
    
    # 计算相关参数
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='使用GPU')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 输出相关参数
    parser.add_argument('--checkpoint_dir', type=str, default='./backend/models/checkpoints',
                        help='检查点保存目录')
    
    args = parser.parse_args()
    
    # 转换为配置字典
    config = vars(args)
    
    # 创建训练器并开始训练
    trainer = FlowerModelTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
