# YOLO OBB预测图片处理系统

基于YOLOv8的定向边界框(OBB)目标检测API系统，提供图片上传、预测和结构化结果返回功能。

## 项目结构

```
interface_yolo/
├── .trae/documents/           # 产品需求和技术架构文档
├── predict/                   # 测试图片目录
├── user/image/YYYYMMDD/      # 用户上传图片按日期存储
├── venv/                     # Python虚拟环境
├── best.pt                   # YOLO模型文件
├── mypredict.py              # 原始YOLO预测脚本
├── predict_api.py            # 完整版API（需要YOLO模型）
├── predict_api_simple.py     # 简化版API（模拟预测，用于测试）
├── test_api.py               # API测试脚本
├── requirements.txt          # Python依赖包
└── README.md                 # 项目说明文档
```

## 功能特性

### 核心功能
1. **图片接收与保存** - 按日期分类存储用户上传的图片
2. **YOLO模型预测** - 调用训练好的模型进行目标检测
3. **OBB格式转换** - 将定向边界框转换为标准坐标格式
4. **API接口** - 提供RESTful API接口
5. **结构化输出** - 返回包含位置、旋转、类型、描述、置信度的JSON结果

### API接口

#### 1. 健康检查
- **URL**: `GET /health`
- **功能**: 检查API服务状态
- **响应**: 
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-27T10:30:00"
}
```

#### 2. 根路径信息
- **URL**: `GET /`
- **功能**: 获取API基本信息
- **响应**:
```json
{
  "message": "YOLO OBB预测API",
  "version": "1.0.0",
  "status": "running",
  "model_loaded": true
}
```

#### 3. 图片预测
- **URL**: `POST /predict`
- **功能**: 上传图片进行目标检测
- **请求**: `multipart/form-data`，包含图片文件
- **响应**:
```json
{
  "status": "success",
  "results": [
    {
      "location": {"x": 100.5, "y": 200.3, "z": 0.0},
      "rotation": {"x": 15.2, "y": -8.7, "z": 0.0},
      "type": 0,
      "description": "floor_tile",
      "confidence": 0.95
    }
  ],
  "image_path": "user/image/20250127/1234567890.jpg",
  "timestamp": "2025-01-27T10:30:00",
  "total_objects": 1
}
```

## 环境要求

- Python 3.8+
- Windows 操作系统
- 至少2GB可用内存

## 安装和运行

### 1. 环境准备

```bash
# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1

# 安装依赖包
pip install -r requirements.txt
```

### 2. 启动API服务

#### 方式一：使用简化版API（推荐用于测试）
```bash
python predict_api_simple.py
```

#### 方式二：使用完整版API（需要YOLO模型）
```bash
python predict_api.py
```

### 3. 访问API

API服务启动后，可通过以下地址访问：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 根路径: http://localhost:8000/

## 测试

### 自动化测试
```bash
python test_api.py
```

### 手动测试
使用Postman或curl测试预测接口：

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@predict/floor_image_182_png.rf.479249a832f34a3f7cb49a401a4bde03.jpg"
```

## 技术栈

- **后端框架**: FastAPI
- **机器学习**: ultralytics YOLO
- **图像处理**: OpenCV, Pillow
- **Web服务器**: Uvicorn
- **数据处理**: NumPy

## 配置说明

### 模型配置
- 模型文件: `best.pt`
- 支持格式: YOLOv8 OBB模型
- 输入格式: 常见图片格式(jpg, png, bmp等)

### 存储配置
- 图片存储路径: `user/image/YYYYMMDD/`
- 日志文件: `api.log`
- 自动创建目录结构

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查`best.pt`文件是否存在
   - 确认模型文件格式正确

2. **依赖包安装失败**
   - 升级pip: `python -m pip install --upgrade pip`
   - 使用CPU版本PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

3. **API启动失败**
   - 检查端口8000是否被占用
   - 确认虚拟环境已激活

### 日志查看
- API运行日志保存在`api.log`文件中
- 控制台会显示实时日志信息

## 开发说明

### 项目架构
- **API服务层**: FastAPI应用，处理HTTP请求
- **业务逻辑层**: 图片处理、模型预测、格式转换
- **模型层**: YOLO模型加载和推理
- **存储层**: 本地文件系统存储

### 扩展开发
1. 添加新的预测模型
2. 支持更多图片格式
3. 添加批量预测功能
4. 集成数据库存储

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目文档: `.trae/documents/`
- 技术支持: 查看项目日志文件