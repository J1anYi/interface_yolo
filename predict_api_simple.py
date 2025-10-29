"""
YOLO OBB预测API接口 - 真实YOLO模型版本
功能：
1. 接收用户上传的图片
2. 保存到本地按日期分类的目录
3. 使用真实YOLO模型进行预测
4. 返回结构化JSON结果

作者：YOLO OBB预测系统
日期：2025-01-27
"""

# 导入必要的库
import os
import io
import json
import math
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# FastAPI相关导入
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入YOLO相关模块
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predict_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
prediction_mode = "real"

# 创建FastAPI应用
app = FastAPI(
    title="YOLO OBB预测API - 真实YOLO模型版",
    description="基于YOLOv8的定向边界框(OBB)目标检测API - 使用真实YOLO模型进行预测",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model() -> bool:
    """加载YOLO模型"""
    global model
    try:
        # 获取脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = "best.pt"
        model_path = os.path.join(script_dir, model_filename)
        
        # 输出调试信息
        current_dir = os.getcwd()
        logger.info(f"当前工作目录: {current_dir}")
        logger.info(f"脚本目录: {script_dir}")
        logger.info(f"模型文件路径: {model_path}")
        
        # 检查脚本目录是否存在且可读
        if not os.path.exists(script_dir):
            logger.error(f"脚本目录不存在: {script_dir}")
            return False
        
        if not os.access(script_dir, os.R_OK):
            logger.error(f"脚本目录无读取权限: {script_dir}")
            return False
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            # 列出脚本目录下的所有文件，帮助调试
            try:
                files_in_dir = os.listdir(script_dir)
                logger.info(f"脚本目录下的文件: {files_in_dir}")
            except Exception as list_error:
                logger.error(f"无法列出脚本目录文件: {list_error}")
            return False
        
        # 检查模型文件权限
        if not os.access(model_path, os.R_OK):
            logger.error(f"模型文件无读取权限: {model_path}")
            # 获取文件权限信息
            try:
                import stat
                file_stat = os.stat(model_path)
                file_mode = stat.filemode(file_stat.st_mode)
                logger.error(f"文件权限: {file_mode}")
                logger.error(f"文件所有者UID: {file_stat.st_uid}")
                logger.error(f"文件组GID: {file_stat.st_gid}")
            except Exception as stat_error:
                logger.error(f"无法获取文件权限信息: {stat_error}")
            return False
        
        # 检查文件大小
        try:
            file_size = os.path.getsize(model_path)
            logger.info(f"模型文件大小: {file_size} 字节")
            if file_size == 0:
                logger.error("模型文件为空")
                return False
        except Exception as size_error:
            logger.error(f"无法获取文件大小: {size_error}")
            return False
        
        logger.info(f"正在加载YOLO模型: {model_path}")
        model = YOLO(model_path)
        logger.info("YOLO模型加载成功")
        return True
    except Exception as e:
        logger.error(f"加载YOLO模型失败: {e}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

def calculate_obb_center_and_rotation(obb_coords: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    计算OBB的中心点和旋转角度（符合UE坐标系统）
    
    Args:
        obb_coords: OBB坐标数组，形状为(4, 2)，包含四个角点的x,y坐标
    
    Returns:
        center: (x, y) 中心点坐标
        rotation: 旋转角度（度），向上为0度，顺时针为正
    """
    # 计算中心点
    center_x = np.mean(obb_coords[:, 0])
    center_y = np.mean(obb_coords[:, 1])
    
    # 计算旋转角度
    # 使用第一条边（从第一个点到第二个点）来计算角度
    dx = obb_coords[1, 0] - obb_coords[0, 0]
    dy = obb_coords[1, 1] - obb_coords[0, 1]
    
    # 计算角度（弧度转度）
    # 标准atan2计算的是相对于水平向右的角度
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # 转换为UE坐标系统：向上为0度，顺时针为正
    # 水平向右为0度 -> 向上为0度：需要减去90度
    # 然后取负值使顺时针为正（因为图像坐标系y轴向下）
    ue_angle = -(angle_deg - 90)
    
    # 将角度规范化到[0, 360)范围
    ue_angle = ue_angle % 360
    
    return (center_x, center_y), ue_angle

def yolo_prediction(image: Image.Image, request_id: str) -> Dict[str, Any]:
    """
    使用真实YOLO模型进行预测
    
    Args:
        image: PIL图像对象
        request_id: 请求ID
    
    Returns:
        预测结果字典
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO模型未加载")
    
    try:
        logger.info(f"[{request_id}] 开始使用真实YOLO模型进行预测")
        
        # 使用YOLO模型进行预测
        results = model.predict(image, verbose=False)
        
        if not results:
            logger.warning(f"[{request_id}] YOLO模型未返回任何结果")
            return []
        
        result = results[0]  # 获取第一个结果
        predictions = []
        
        # 检查是否有OBB结果
        if hasattr(result, 'obb') and result.obb is not None:
            logger.info(f"[{request_id}] 检测到OBB格式结果")
            
            # 处理OBB结果
            obb_data = result.obb
            boxes = obb_data.xyxyxyxy.cpu().numpy()  # OBB坐标
            confidences = obb_data.conf.cpu().numpy()  # 置信度
            classes = obb_data.cls.cpu().numpy()  # 类别
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                # 计算OBB中心点和旋转角度
                center, rotation = calculate_obb_center_and_rotation(box)
                
                # 获取类别名称
                class_name = model.names[int(cls)]
                
                prediction = {
                    "location": {
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "z": 0.0
                    },
                    "rotation": {
                        "x": 0.0,  # 滚转角，设为0
                        "y": 0.0,  # 俯仰角，设为0
                        "z": float(rotation)  # 旋转角，向上为0度
                    },
                    "type": int(cls),
                    "obj_name": class_name,
                    "confidence": float(conf)
                }
                predictions.append(prediction)
                
                logger.info(f"[{request_id}] OBB检测对象 {i+1}: {class_name}, 置信度: {conf:.3f}, 中心: ({center[0]:.2f}, {center[1]:.2f}), 旋转: {rotation:.2f}°")
        
        # 如果没有OBB结果，检查是否有标准边界框结果
        elif hasattr(result, 'boxes') and result.boxes is not None:
            logger.info(f"[{request_id}] 检测到标准边界框格式结果")
            
            boxes_data = result.boxes
            boxes = boxes_data.xyxy.cpu().numpy()  # 边界框坐标
            confidences = boxes_data.conf.cpu().numpy()  # 置信度
            classes = boxes_data.cls.cpu().numpy()  # 类别
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                # 计算边界框中心点
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 获取类别名称
                class_name = model.names[int(cls)]
                
                prediction = {
                    "location": {
                        "x": float(center_x),
                        "y": float(center_y),
                        "z": 0.0
                    },
                    "rotation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    },
                    "type": int(cls),
                    "obj_name": class_name,
                    "confidence": float(conf)
                }
                predictions.append(prediction)
                
                logger.info(f"[{request_id}] 标准框检测对象 {i+1}: {class_name}, 置信度: {conf:.3f}, 中心: ({center_x:.2f}, {center_y:.2f})")
        
        else:
            logger.warning(f"[{request_id}] 未检测到任何对象")
        
        logger.info(f"[{request_id}] 真实YOLO预测完成，检测到 {len(predictions)} 个对象")
        return predictions
        
    except Exception as e:
        logger.error(f"[{request_id}] 真实YOLO预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    global prediction_mode
    
    logger.info("============================================================")
    logger.info("正在启动YOLO OBB预测API（真实YOLO模型版）...")
    
    # 强制加载YOLO模型
    if load_model():
        prediction_mode = "real"
        logger.info("YOLO模型加载成功，使用真实预测模式")
    else:
        logger.error("YOLO模型加载失败，API无法启动")
        raise RuntimeError("YOLO模型加载失败，请检查模型文件")
    
    logger.info(f"当前预测模式: {prediction_mode}")
    logger.info("服务地址: http://0.0.0.0:8000")
    logger.info("API文档: http://0.0.0.0:8000/docs")
    logger.info("============================================================")

@app.get("/")
async def root():
    """根路径，返回API基本信息"""
    return {
        "message": "YOLO OBB预测API - 真实YOLO模型版",
        "version": "4.0.0",
        "status": "running",
        "mode": prediction_mode,
        "model_loaded": model is not None,
        "yolo_available": True,
        "description": f"当前使用{prediction_mode}预测模式，真实YOLO模型预测"
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "yolo_available": True,
        "prediction_mode": prediction_mode
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    预测接口
    
    Args:
        file: 上传的图片文件
    
    Returns:
        预测结果
    """
    start_time = time.time()
    
    # 生成请求ID
    timestamp = int(time.time() * 1000)
    request_id = f"req_{timestamp}"
    
    logger.info(f"[{request_id}] 收到预测请求，文件名: {file.filename}")
    
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            # 如果content_type为空或不是图片，尝试从文件名判断
            if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
                raise HTTPException(status_code=400, detail="请上传图片文件")
        
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"[{request_id}] 图片加载成功，尺寸: {image.size}")
        
        # 保存上传的图片
        date_folder = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("user", "image", date_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成唯一文件名
        file_hash = hashlib.md5(image_data).hexdigest()[:8]
        save_filename = f"{timestamp}_{file_hash}.jpg"
        save_path = os.path.join(save_dir, save_filename)
        
        # 保存图片
        image.save(save_path, "JPEG", quality=95)
        logger.info(f"[{request_id}] 图片已保存到: {save_path}")
        
        # 使用真实YOLO模型进行预测
        logger.info(f"[{request_id}] 开始真实YOLO预测")
        predictions = yolo_prediction(image, request_id)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建响应
        response = {
            "functionName": "make_home",
            "status": "success",
            "results": predictions,
            "image_path": save_path.replace("\\", "/"),
            "timestamp": datetime.now().isoformat(),
            "total_objects": len(predictions),
            "mode": prediction_mode,
            "request_id": request_id,
            "model_info": {
                "model_classes": model.names if model else {},
                "model_loaded": model is not None,
                "yolo_available": True,
                "model_path": "best.pt",
                "prediction_note": "使用真实YOLO模型进行目标检测"
            },
            "processing_time_seconds": round(processing_time, 3)
        }
        
        logger.info(f"[{request_id}] 预测完成，检测到 {len(predictions)} 个对象，耗时: {processing_time:.3f}秒")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] 预测过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("user/image", exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        "predict_api_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )