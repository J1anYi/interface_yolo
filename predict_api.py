"""
YOLO OBB预测API接口
功能：
1. 接收用户上传的图片
2. 保存到本地按日期分类的目录
3. 调用YOLO模型进行预测
4. 转换OBB格式为标准坐标
5. 返回结构化JSON结果

作者：YOLO OBB预测系统
日期：2025-01-27
"""

import os
import cv2
import json
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="YOLO OBB预测API",
    description="基于YOLOv8的定向边界框(OBB)目标检测API",
    version="1.0.0"
)

# 全局变量
model = None

def load_model():
    """加载YOLO模型"""
    global model
    try:
        # 检查模型文件是否存在
        model_path = "best.pt"
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = YOLO(model_path)
        logger.info(f"成功加载YOLO模型: {model_path}")
        return True
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return False

def save_uploaded_image(img_file: UploadFile) -> str:
    """
    保存上传的图片到按日期分类的目录
    
    Args:
        img_file: 上传的图片文件
        
    Returns:
        str: 保存的文件路径
    """
    try:
        # 创建按日期分类的目录
        today = datetime.now().strftime("%Y%m%d")
        save_dir = Path(f"user/image/{today}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        timestamp = int(time.time())
        file_extension = img_file.filename.split('.')[-1] if img_file.filename else 'jpg'
        save_path = save_dir / f"{timestamp}.{file_extension}"
        
        # 保存文件
        with open(save_path, "wb") as f:
            content = img_file.file.read()
            f.write(content)
        
        logger.info(f"图片保存成功: {save_path}")
        return str(save_path)
        
    except Exception as e:
        logger.error(f"保存图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存图片失败: {str(e)}")

def calculate_obb_center_and_rotation(obb_points: np.ndarray) -> tuple:
    """
    计算OBB的中心点和旋转角度
    
    Args:
        obb_points: OBB的四个顶点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        tuple: (center_x, center_y, rotation_x, rotation_y)
    """
    try:
        # 计算中心点
        center_x = float(np.mean(obb_points[:, 0]))
        center_y = float(np.mean(obb_points[:, 1]))
        
        # 计算旋转向量（使用第一条边的方向）
        edge_vector = obb_points[1] - obb_points[0]
        rotation_x = float(edge_vector[0])
        rotation_y = float(edge_vector[1])
        
        return center_x, center_y, rotation_x, rotation_y
        
    except Exception as e:
        logger.error(f"计算OBB中心和旋转失败: {str(e)}")
        # 返回默认值
        return 0.0, 0.0, 0.0, 0.0

def predict_and_convert(image_path: str) -> List[Dict[str, Any]]:
    """
    使用YOLO模型进行预测并转换OBB格式
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        List[Dict]: 预测结果列表
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        # 进行预测
        results = model.predict(
            source=image_path,
            save=False,
            show=False,
            verbose=False
        )
        
        output = []
        
        # 处理预测结果
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # 处理OBB结果
                boxes = result.obb.xyxyxyxy.cpu().numpy()  # 获取OBB坐标
                confidences = result.obb.conf.cpu().numpy()  # 获取置信度
                classes = result.obb.cls.cpu().numpy()  # 获取类别
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    # 计算中心点和旋转
                    center_x, center_y, rotation_x, rotation_y = calculate_obb_center_and_rotation(box)
                    
                    # 获取类别名称
                    class_name = model.names.get(int(cls), f"class_{int(cls)}")
                    
                    obj_result = {
                        "location": {
                            "x": center_x,
                            "y": center_y,
                            "z": 0.0
                        },
                        "rotation": {
                            "x": rotation_x,
                            "y": rotation_y,
                            "z": 0.0
                        },
                        "type": int(cls),
                        "description": class_name,
                        "confidence": float(conf)
                    }
                    output.append(obj_result)
                    
            elif hasattr(result, 'boxes') and result.boxes is not None:
                # 如果没有OBB，处理普通边界框
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
                confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
                classes = result.boxes.cls.cpu().numpy()  # 获取类别
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    # 计算中心点
                    center_x = float((box[0] + box[2]) / 2)
                    center_y = float((box[1] + box[3]) / 2)
                    
                    # 获取类别名称
                    class_name = model.names.get(int(cls), f"class_{int(cls)}")
                    
                    obj_result = {
                        "location": {
                            "x": center_x,
                            "y": center_y,
                            "z": 0.0
                        },
                        "rotation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0
                        },
                        "type": int(cls),
                        "description": class_name,
                        "confidence": float(conf)
                    }
                    output.append(obj_result)
        
        logger.info(f"预测完成，检测到 {len(output)} 个对象")
        return output
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    logger.info("正在启动YOLO OBB预测API...")
    if not load_model():
        logger.error("模型加载失败，API可能无法正常工作")
    else:
        logger.info("API启动成功")

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "YOLO OBB预测API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    """
    预测接口
    
    Args:
        file: 用户上传的图片文件
        
    Returns:
        JSON格式的预测结果
    """
    try:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="请上传图片文件")
        
        logger.info(f"接收到预测请求，文件名: {file.filename}")
        
        # 保存图片
        img_path = save_uploaded_image(file)
        
        # 预测并转换格式
        results = predict_and_convert(img_path)
        
        response = {
            "status": "success",
            "results": results,
            "image_path": img_path,
            "timestamp": datetime.now().isoformat(),
            "total_objects": len(results)
        }
        
        logger.info(f"预测成功，返回 {len(results)} 个检测结果")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测接口错误: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    # 启动API服务
    uvicorn.run(
        "predict_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )