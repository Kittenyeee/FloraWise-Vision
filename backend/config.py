"""Flask应用配置文件"""
import os
from datetime import timedelta


class Config:
    """基础配置"""
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'florawise-vision-secret-key-2026'
    DEBUG = False
    TESTING = False
    
    # 文件上传配置
    UPLOAD_FOLDER = 'backend/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    
    # 模型配置
    MODEL_PATH = 'backend/models/model_weights.pth'
    CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    INPUT_SIZE = 224
    
    # 推理配置
    CONFIDENCE_THRESHOLD = 0.5
    DEVICE = 'cuda' if os.environ.get('USE_GPU') else 'cpu'
    
    # 会话配置
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/app.log'


class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    UPLOAD_FOLDER = 'backend/test_uploads'
    MODEL_PATH = 'backend/models/test_model.pth'


# 配置映射
config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """获取配置对象"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config_dict.get(env, config_dict['default'])
