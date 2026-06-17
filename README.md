# FloraWise Vision - 花卉分类智能助手

## 📌 项目简介
基于深度学习的花卉智能分类系统，支持多种输入方式（本地上传、实时拍摄），实现高精度的花卉品种识别。

**支持分类**：菊花 (Daisy) | 蒲公英 (Dandelion) | 玫瑰 (Roses) | 向日葵 (Sunflowers) | 郁金香 (Tulips)

## 📊 项目目标
1. ✅ 搭建完整的图像分类识别系统（数据集→训练→推理→展示）
2. ✅ 构建高质量花卉数据集（5个品种，包含数据增强）
3. ✅ 开发轻量化深度学习模型（MobileNetV2 + 微调）
4. ✅ 开发Web交互前端界面（Flask后端 + Vue3前端）

## 📁 项目结构
```
FloraWise-Vision/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── models/
│   ├── utils/
│   └── routes/
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── data_analysis.py
└── docs/
```
