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
├── backend/                    # 后端应用
│   ├── app.py                 # Flask应用入口
│   ├── config.py              # 配置文件
│   ├── requirements.txt        # 依赖清单
│   ├── models/
│   │   ├── __init__.py
│   │   ├── flower_model.py    # 模型定义
│   │   └── model_weights.pth  # 预训练权重
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # 数据加载器
│   │   ├── preprocessing.py   # 数据预处理
│   │   └── inference.py       # 推理引擎
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py             # API路由
│   └── uploads/               # 上传文件临时目录
│
├── frontend/                   # 前端应用
│   ├── index.html             # 主页面
│   ├── css/
│   │   └── style.css          # 样式表
│   └── js/
│       └── app.js             # 交互逻辑
│
├── training/                   # 训练脚本
│   ├── train.py               # 训练主程序
│   ├── evaluate.py            # 模型评估
│   └── data_analysis.py       # 数据分析
│
├── docs/                       # 文档
│   └── 课程设计报告.md
│
└── README.md                   # 项目说明（本文件）
```

## 🛠 技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| **后端框架** | Flask | 2.3+ |
| **深度学习** | PyTorch | 2.0+ |
| **前端框架** | Vue 3 + Bootstrap 5 | - |
| **数据处理** | OpenCV + NumPy | - |
| **数据库** | SQLite | 3.0+ |

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/Kittenyeee/FloraWise-Vision.git
cd FloraWise-Vision

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r backend/requirements.txt
```

### 2. 数据准备
```bash
# 数据集结构（flower_data）
flower_data/
├── train/
│   ├── daisy/
│   ├── dandelion/
│   ├── roses/
│   ├── sunflowers/
│   └── tulips/
└── val/
    ├── daisy/
    ├── dandelion/
    ├── roses/
    ├── sunflowers/
    └── tulips/
```

### 3. 模型训练
```bash
python training/train.py \
    --data_path ./flower_data \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

### 4. 启动应用
```bash
# 启动后端
python backend/app.py

# 在浏览器中打开
http://localhost:5000
```

## 📈 模型性能

| 指标 | 值 |
|------|-----|
| Top-1 准确率 | ~92% |
| 推理时间 | ~50ms/张 |
| 模型大小 | ~14MB |

## 👥 小组分工

| 成员 | 学号 | 主要任务 | 权重 |
|------|------|--------|------|
| 成员1 | XXXXX | 数据处理、模型训练、模型评估 | 50% |
| 成员2 | XXXXX | 前端界面设计、后端API、前后端集成 | 50% |

## 📝 关键特性

✨ **智能识别**
- 支持5种花卉品种识别
- 高精度深度学习模型
- 置信度分数输出

🖥️ **友好交互**
- 拖拽上传图片
- 实时预览上传图像
- 详细的分类结果展示
- 响应式网页设计

⚡ **高效部署**
- 轻量化模型（14MB）
- GPU/CPU双支持
- 快速推理速度

## 🔍 测试覆盖

- ✅ 单元测试：数据加载、图像预处理
- ✅ 集成测试：模型推理、API接口
- ✅ 端到端测试：完整的用户交互流程

## 📦 依赖清单
详见 `backend/requirements.txt`

## 🐛 常见问题

**Q: 如何更换模型？**
A: 修改 `backend/models/flower_model.py`，支持ResNet、EfficientNet等预训练模型。

**Q: 如何提升准确率？**
A: 增加训练数据、调整超参数、使用数据增强技术。

**Q: 如何部署到云服务器？**
A: 使用 Docker 容器化，详见 `Dockerfile`。

## 📄 许可证
MIT License

## 📞 联系方式
Issues: https://github.com/Kittenyeee/FloraWise-Vision/issues

---

**最后更新**：2026年6月
**项目状态**：开发中 🚀
