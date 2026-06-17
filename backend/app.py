"""
Flask应用主文件
创建和配置 Flask 应用
"""
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
import logging
import torch
import os
from dotenv import load_dotenv

from config import get_config
from models.flower_model import FlowerClassifier
from utils.inference import InferenceEngine
from routes.api import create_api_blueprint

# 加载环境变量
load_dotenv()


def create_app(config_name: str = None) -> Flask:
    """应用工厂函数
    
    Args:
        config_name: 配置名称 ('development', 'production', 'testing')
        
    Returns:
        Flask 应用实例
    """
    
    # 创建 Flask 应用
    app = Flask(__name__, 
                template_folder='../frontend',
                static_folder='../frontend')
    
    # 加载配置
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    config = get_config(config_name)
    app.config.from_object(config)
    
    # 启用 CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # 创建上传目录
    upload_dir = Path(app.config['UPLOAD_FOLDER'])
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    setup_logging(app)
    
    # 初始化模型和推理引擎
    try:
        app.logger.info('正在加载模型...')
        inference_engine = initialize_model(app.config)
        app.inference_engine = inference_engine
        app.logger.info('模型加载成功')
    except Exception as e:
        app.logger.error(f'模型加载失败: {str(e)}')
        raise
    
    # 注册 API 蓝图
    api_bp = create_api_blueprint(inference_engine)
    app.register_blueprint(api_bp)
    
    # 注册前端路由
    register_frontend_routes(app)
    
    # 全局错误处理器
    register_error_handlers(app)
    
    return app


def initialize_model(config: dict) -> InferenceEngine:
    """创建模型和推理引擎
    
    Args:
        config: 配置字典
        
    Returns:
        InferenceEngine 实例
    """
    
    # 检查模型文件是否存在
    model_path = Path(config['MODEL_PATH'])
    if not model_path.exists():
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    
    # 设置设备
    device = config['DEVICE']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载模型
    model = FlowerClassifier(num_classes=len(config['CLASS_NAMES']), pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的检查点格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 创建推理引擎
    inference_engine = InferenceEngine(
        model=model,
        class_names=config['CLASS_NAMES'],
        device=device
    )
    
    return inference_engine


def register_frontend_routes(app: Flask) -> None:
    """注册前端路由
    
    Args:
        app: Flask 应用
    """
    
    @app.route('/')
    def index():
        """主页面"""
        return render_template('index.html')
    
    @app.route('/assets/<path:filename>')
    def serve_assets(filename):
        """提供静态资源"""
        return send_from_directory('frontend/assets', filename)


def register_error_handlers(app: Flask) -> None:
    """注册全局错误处理器
    
    Args:
        app: Flask 应用
    """
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'success': False,
            'error': '页面不存在'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500


def setup_logging(app: Flask) -> None:
    """配置日志
    
    Args:
        app: Flask 应用
    """
    
    # 创建 logs 目录
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # 配置日志处理器
    log_file = logs_dir / 'app.log'
    
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)


if __name__ == '__main__':
    # 创建应用
    app = create_app()
    
    # 运行应用
    app.run(
        host=os.environ.get('API_HOST', '0.0.0.0'),
        port=int(os.environ.get('API_PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
