"""
API路由模块
定义所有的Flask路由和API端点
"""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import torch
from typing import Dict, Any
import time

api_bp = Blueprint('api', __name__, url_prefix='/api')


def create_api_blueprint(inference_engine):
    """创建带有推理引擎的API蓝图
    
    Args:
        inference_engine: InferenceEngine对象
        
    Returns:
        Flask Blueprint应用
    """
    
    @api_bp.route('/health', methods=['GET'])
    def health_check():
        """健康检查端点"""
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'device': str(inference_engine.device)
        }), 200
    
    @api_bp.route('/model/info', methods=['GET'])
    def get_model_info():
        """获取模型信息"""
        try:
            model_info = inference_engine.get_model_info()
            return jsonify({
                'success': True,
                'data': model_info
            }), 200
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @api_bp.route('/predict', methods=['POST'])
    def predict_image():
        """推理端点 - 支持文件上传或Base64编码的图像
        
        上传方式1: form-data
            - file: 图像文件
            - confidence_threshold: 置信度阈值 (可选, 默认0.0)
            
        上传方式2: JSON
            - image_base64: Base64编码的图像数据
            - confidence_threshold: 置信度阈值 (可选)
        """
        try:
            confidence_threshold = float(request.form.get('confidence_threshold', 0.0))
            
            # 正常文件上传
            if 'file' in request.files:
                file = request.files['file']
                
                # 检查文件不为None且文件名不为空
                if file.filename == '':
                    return jsonify({
                        'success': False,
                        'error': '没有选择文件'
                    }), 400
                
                # 检查文件类型
                if not allowed_file(file.filename):
                    return jsonify({
                        'success': False,
                        'error': f'不支持的文件类型: {file.filename}'
                    }), 400
                
                # 保存文件
                filename = secure_filename(file.filename)
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{filename}"
                
                upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
                upload_dir.mkdir(parents=True, exist_ok=True)
                file_path = upload_dir / filename
                file.save(str(file_path))
                
                # 执行推理
                result = inference_engine.predict(str(file_path), confidence_threshold)
                
                # 清理上传的文件
                try:
                    os.remove(str(file_path))
                except:
                    pass
                
                return jsonify({
                    'success': result['success'],
                    'data': result if result['success'] else None,
                    'error': result.get('error', None)
                }), 200 if result['success'] else 400
            
            # Base64编码输入
            elif request.is_json:
                data = request.get_json()
                if 'image_base64' not in data:
                    return jsonify({
                        'success': False,
                        'error': '不提供image_base64'
                    }), 400
                
                import base64
                import io
                from PIL import Image
                
                try:
                    image_data = base64.b64decode(data['image_base64'])
                    image = Image.open(io.BytesIO(image_data))
                    
                    result = inference_engine.predict(image, confidence_threshold)
                    return jsonify({
                        'success': result['success'],
                        'data': result if result['success'] else None,
                        'error': result.get('error', None)
                    }), 200 if result['success'] else 400
                
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'Base64解码失败: {str(e)}'
                    }), 400
            
            else:
                return jsonify({
                    'success': False,
                    'error': '不支持的请求方式'
                }), 400
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'推理错误: {str(e)}'
            }), 500
    
    @api_bp.route('/classes', methods=['GET'])
    def get_classes():
        """获取支持的类别列表"""
        try:
            classes = inference_engine.get_class_names()
            return jsonify({
                'success': True,
                'classes': classes,
                'num_classes': len(classes)
            }), 200
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @api_bp.errorhandler(400)
    def bad_request(error):
        """处理400错误"""
        return jsonify({
            'success': False,
            'error': '错误的请求'
        }), 400
    
    @api_bp.errorhandler(404)
    def not_found(error):
        """处理404错误"""
        return jsonify({
            'success': False,
            'error': '端点不存在'
        }), 404
    
    @api_bp.errorhandler(500)
    def internal_error(error):
        """处理500错误"""
        return jsonify({
            'success': False,
            'error': '服务器内部错误'
        }), 500
    
    return api_bp


def allowed_file(filename: str) -> bool:
    """检查文件是否是允许的图像文件
    
    Args:
        filename: 文件名
        
    Returns:
        是否允许
    """
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
