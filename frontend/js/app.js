/**
 * FloraWise Vision - 前端交互JavaScript
 * 处理图像上传、推理请求、结果显示
 */

// API基地址
const API_BASE = '/api';

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const clearBtn = document.getElementById('clearBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const uploadPlaceholder = uploadArea.querySelector('.upload-placeholder');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultContainer = document.getElementById('resultContainer');
const emptyState = document.getElementById('emptyState');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// 状态
let selectedFile = null;
let isProcessing = false;

/**
 * 初始化事件监听
 */
function initEventListeners() {
    // 上传区域拖拽事件
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // 文件输入变化
    imageInput.addEventListener('change', handleFileSelect);
    
    // 按钮点击事件
    uploadBtn.addEventListener('click', handleUpload);
    clearBtn.addEventListener('click', clearPreview);
    
    // 页面加载时检查后端
    checkBackendHealth();
}

/**
 * 处理拖拽覆盖
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
}

/**
 * 处理拖拽离开
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

/**
 * 处理拖拽放下
 */
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isImageFile(file)) {
            selectedFile = file;
            displayPreview(file);
        } else {
            showError('请选择图像文件 (JPG, PNG, GIF, BMP)');
        }
    }
}

/**
 * 处理文件选择
 */
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        const file = files[0];
        if (isImageFile(file)) {
            selectedFile = file;
            displayPreview(file);
        } else {
            showError('请选择图像文件 (JPG, PNG, GIF, BMP)');
            imageInput.value = '';
        }
    }
}

/**
 * 检查是否是图像文件
 */
function isImageFile(file) {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp'];
    return allowedTypes.includes(file.type);
}

/**
 * 显示图像预览
 */
function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        fileName.textContent = `文件: ${file.name} (${formatFileSize(file.size)})`;
        
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
        uploadBtn.disabled = false;
        clearBtn.style.display = 'inline-block';
        
        // 清除之前的结果
        clearResults();
    };
    
    reader.readAsDataURL(file);
}

/**
 * 清除预览
 */
function clearPreview() {
    selectedFile = null;
    imageInput.value = '';
    imagePreview.style.display = 'none';
    uploadPlaceholder.style.display = 'block';
    uploadBtn.disabled = true;
    clearBtn.style.display = 'none';
    clearResults();
}

/**
 * 处理上传和推理
 */
async function handleUpload() {
    if (!selectedFile) {
        showError('请先选择图像');
        return;
    }
    
    if (isProcessing) {
        return;
    }
    
    isProcessing = true;
    uploadBtn.disabled = true;
    
    try {
        // 显示加载状态
        emptyState.style.display = 'none';
        resultContainer.style.display = 'none';
        errorMessage.style.display = 'none';
        loadingSpinner.style.display = 'block';
        
        // 创建FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('confidence_threshold', '0.0');
        
        // 发送请求
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // 隐藏加载状态
        loadingSpinner.style.display = 'none';
        
        if (data.success && data.data) {
            displayResults(data.data);
        } else {
            showError(data.error || '推理失败，请重试');
        }
    } catch (error) {
        loadingSpinner.style.display = 'none';
        showError(`请求错误: ${error.message}`);
        console.error('推理错误:', error);
    } finally {
        isProcessing = false;
        uploadBtn.disabled = false;
    }
}

/**
 * 显示推理结果
 */
function displayResults(result) {
    try {
        // 显示结果容器
        emptyState.style.display = 'none';
        resultContainer.style.display = 'block';
        errorMessage.style.display = 'none';
        
        // 显示顶级分类结果
        const topClass = document.getElementById('topClass');
        const confidenceScore = document.getElementById('confidenceScore');
        const confidenceBar = document.getElementById('confidenceBar');
        
        topClass.textContent = formatClassName(result.top_class);
        const confidence = result.confidence * 100;
        confidenceScore.textContent = confidence.toFixed(2) + '%';
        confidenceBar.style.width = confidence + '%';
        
        // 显示所有概率
        const probabilityItems = document.getElementById('probabilityItems');
        probabilityItems.innerHTML = '';
        
        if (result.all_predictions && Array.isArray(result.all_predictions)) {
            result.all_predictions.forEach((pred, index) => {
                const itemDiv = createProbabilityItem(pred);
                probabilityItems.appendChild(itemDiv);
            });
        }
        
        // 添加淡入动画
        resultContainer.classList.add('fade-in');
    } catch (error) {
        console.error('显示结果错误:', error);
        showError('显示结果时出错');
    }
}

/**
 * 创建概率项目元素
 */
function createProbabilityItem(prediction) {
    const div = document.createElement('div');
    div.className = 'probability-item';
    
    const confidence = prediction.confidence * 100;
    const labelColor = getColorByConfidence(confidence);
    
    div.innerHTML = `
        <span class="item-label">${formatClassName(prediction.class)}</span>
        <div class="item-bar">
            <div class="item-bar-fill" style="width: ${confidence}%;"></div>
        </div>
        <span class="item-value">${confidence.toFixed(1)}%</span>
    `;
    
    return div;
}

/**
 * 根据置信度获取颜色
 */
function getColorByConfidence(confidence) {
    if (confidence >= 80) return '#10b981';
    if (confidence >= 60) return '#6366f1';
    if (confidence >= 40) return '#f59e0b';
    return '#ef4444';
}

/**
 * 格式化类名
 */
function formatClassName(className) {
    const classMap = {
        'daisy': '菊花',
        'dandelion': '蒲公英',
        'roses': '玫瑰',
        'sunflowers': '向日葵',
        'tulips': '郁金香'
    };
    return classMap[className.toLowerCase()] || className;
}

/**
 * 清除结果
 */
function clearResults() {
    resultContainer.style.display = 'none';
    emptyState.style.display = 'block';
    errorMessage.style.display = 'none';
    loadingSpinner.style.display = 'none';
}

/**
 * 显示错误信息
 */
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    resultContainer.style.display = 'none';
    emptyState.style.display = 'none';
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * 检查后端健康状态
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            console.warn('后端可能未就绪');
        } else {
            const data = await response.json();
            console.log('后端状态:', data);
        }
    } catch (error) {
        console.error('无法连接到后端:', error);
    }
}

/**
 * 页面加载完成后初始化
 */
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    console.log('FloraWise Vision 已就绪！');
});
