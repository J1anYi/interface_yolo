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
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入YOLO相关模块
import numpy as np
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

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
wall_model = None
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

def load_wall_model() -> bool:
    """加载墙体YOLO模型（best_wall.pt）"""
    global wall_model
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_filename = "best_wall.pt"
        model_path = os.path.join(script_dir, model_filename)

        current_dir = os.getcwd()
        logger.info(f"当前工作目录: {current_dir}")
        logger.info(f"脚本目录: {script_dir}")
        logger.info(f"墙体模型文件路径: {model_path}")

        if not os.path.exists(model_path):
            logger.error(f"墙体模型文件不存在: {model_path}")
            try:
                files_in_dir = os.listdir(script_dir)
                logger.info(f"脚本目录下的文件: {files_in_dir}")
            except Exception as list_error:
                logger.error(f"无法列出脚本目录文件: {list_error}")
            return False

        if not os.access(model_path, os.R_OK):
            logger.error(f"墙体模型文件无读取权限: {model_path}")
            return False

        try:
            file_size = os.path.getsize(model_path)
            logger.info(f"墙体模型文件大小: {file_size} 字节")
            if file_size == 0:
                logger.error("墙体模型文件为空")
                return False
        except Exception as size_error:
            logger.error(f"无法获取墙体模型文件大小: {size_error}")
            return False

        logger.info(f"正在加载墙体YOLO模型: {model_path}")
        wall_model = YOLO(model_path)
        logger.info("墙体YOLO模型加载成功")
        return True
    except Exception as e:
        logger.error(f"加载墙体YOLO模型失败: {e}")
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

def calculate_wall_endpoints_from_obb(obb_coords: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    根据OBB四点坐标，计算两条较短边的中点（墙的起止点）

    返回 (start_point, end_point)，每个为 (x, y)
    """
    p = obb_coords.reshape(4, 2)
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    d01 = dist(p[0], p[1])
    d12 = dist(p[1], p[2])
    d23 = dist(p[2], p[3])
    d30 = dist(p[3], p[0])

    # 判断哪一对对边更短（两条相对边长度相等）
    # 对边成对：(0-1, 2-3) 与 (1-2, 3-0)
    pair1 = (d01, d23)
    pair2 = (d12, d30)
    use_pair1 = (d01 + d23) <= (d12 + d30)

    if use_pair1:
        mid1 = ((p[0][0] + p[1][0]) / 2.0, (p[0][1] + p[1][1]) / 2.0)
        mid2 = ((p[2][0] + p[3][0]) / 2.0, (p[2][1] + p[3][1]) / 2.0)
    else:
        mid1 = ((p[1][0] + p[2][0]) / 2.0, (p[1][1] + p[2][1]) / 2.0)
        mid2 = ((p[3][0] + p[0][0]) / 2.0, (p[3][1] + p[0][1]) / 2.0)

    return mid1, mid2

def _is_close_rel(a: float, b: float, tol_ratio: float = 0.02) -> bool:
    """相对误差判定，98%相似度等价于相对误差<=2%"""
    try:
        return abs(a - b) <= tol_ratio * max(abs(a), abs(b), 1.0)
    except Exception:
        return False

def _is_points_close(p1: Dict[str, Any], p2: Dict[str, Any], tol_ratio: float = 0.02) -> bool:
    """端点近似重合判定：x、y均满足相对误差<=2%"""
    return (
        isinstance(p1, dict) and isinstance(p2, dict) and
        _is_close_rel(float(p1.get("x", 0.0)), float(p2.get("x", 0.0)), tol_ratio) and
        _is_close_rel(float(p1.get("y", 0.0)), float(p2.get("y", 0.0)), tol_ratio)
    )

def _segment_orientation(sp: Dict[str, Any], ep: Dict[str, Any], tol_ratio: float = 0.02) -> Optional[str]:
    """墙段方向判断：共享相同x为竖直，共享相同y为水平，否则返回None"""
    sx, sy = float(sp.get("x", 0.0)), float(sp.get("y", 0.0))
    ex, ey = float(ep.get("x", 0.0)), float(ep.get("y", 0.0))
    if _is_close_rel(sx, ex, tol_ratio):
        return "vertical"
    if _is_close_rel(sy, ey, tol_ratio):
        return "horizontal"
    return None

def _line_value(sp: Dict[str, Any], ep: Dict[str, Any], orientation: Optional[str]) -> Optional[float]:
    """同一直线判定用的线值：竖直用x均值，水平用y均值"""
    if orientation == "vertical":
        return (float(sp.get("x", 0.0)) + float(ep.get("x", 0.0))) / 2.0
    if orientation == "horizontal":
        return (float(sp.get("y", 0.0)) + float(ep.get("y", 0.0))) / 2.0
    return None

def _try_merge_wall(a: Dict[str, Any], b: Dict[str, Any], tol_ratio: float = 0.02, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """尝试按规则合并两个墙段，返回新墙或None"""
    pa = a.get("points") or {}
    pb = b.get("points") or {}
    spA, epA = pa.get("start_point"), pa.get("end_point")
    spB, epB = pb.get("start_point"), pb.get("end_point")
    if not (isinstance(spA, dict) and isinstance(epA, dict) and isinstance(spB, dict) and isinstance(epB, dict)):
        return None

    oriA = _segment_orientation(spA, epA, tol_ratio)
    oriB = _segment_orientation(spB, epB, tol_ratio)
    if oriA is None or oriB is None or oriA != oriB:
        return None

    lvA = _line_value(spA, epA, oriA)
    lvB = _line_value(spB, epB, oriB)
    if lvA is None or lvB is None or not _is_close_rel(lvA, lvB, tol_ratio):
        return None

    # 端点连接判定（4种组合）
    if _is_points_close(spA, spB, tol_ratio):
        new_start, new_end = epA, epB
        loc_x = (float(spA["x"]) + float(spB["x"])) / 2.0
        loc_y = (float(spA["y"]) + float(spB["y"])) / 2.0
    elif _is_points_close(spA, epB, tol_ratio):
        new_start, new_end = epA, spB
        loc_x = (float(spA["x"]) + float(epB["x"])) / 2.0
        loc_y = (float(spA["y"]) + float(epB["y"])) / 2.0
    elif _is_points_close(epA, spB, tol_ratio):
        new_start, new_end = spA, epB
        loc_x = (float(epA["x"]) + float(spB["x"])) / 2.0
        loc_y = (float(epA["y"]) + float(spB["y"])) / 2.0
    elif _is_points_close(epA, epB, tol_ratio):
        new_start, new_end = spA, spB
        loc_x = (float(epA["x"]) + float(epB["x"])) / 2.0
        loc_y = (float(epA["y"]) + float(epB["y"])) / 2.0
    else:
        return None

    rot_z = 0.0 if oriA == "vertical" else 90.0
    confA = float(a.get("confidence", 0.0))
    confB = float(b.get("confidence", 0.0))
    new_conf = min(confA, confB)

    merged = {
        "location": {"x": float(loc_x), "y": float(loc_y), "z": 0.0},
        "points": {
            "start_point": {"x": float(new_start["x"]), "y": float(new_start["y"]), "z": 0.0},
            "end_point": {"x": float(new_end["x"]), "y": float(new_end["y"]), "z": 0.0}
        },
        "rotation": {"x": 0.0, "y": 0.0, "z": float(rot_z)},
        "type": 16,
        "obj_name": "wall",
        "confidence": float(new_conf),
        "status": "已读"
    }
    # 详细日志输出
    try:
        locA = a.get("location") or {}
        locB = b.get("location") or {}
        msg_prefix = f"[{request_id}] " if request_id else ""
        logger.info(
            msg_prefix +
            (
                "墙体合并: "
                f"A(start=({float(spA['x']):.2f},{float(spA['y']):.2f}), end=({float(epA['x']):.2f},{float(epA['y']):.2f}), "
                f"pos=({float(locA.get('x', 0.0)):.2f},{float(locA.get('y', 0.0)):.2f})) + "
                f"B(start=({float(spB['x']):.2f},{float(spB['y']):.2f}), end=({float(epB['x']):.2f},{float(epB['y']):.2f}), "
                f"pos=({float(locB.get('x', 0.0)):.2f},{float(locB.get('y', 0.0)):.2f})) -> "
                f"新墙(start=({float(merged['points']['start_point']['x']):.2f},{float(merged['points']['start_point']['y']):.2f}), "
                f"end=({float(merged['points']['end_point']['x']):.2f},{float(merged['points']['end_point']['y']):.2f}), "
                f"pos=({float(merged['location']['x']):.2f},{float(merged['location']['y']):.2f}))"
            )
        )
    except Exception as log_e:
        logger.error(f"{msg_prefix if request_id else ''}墙体合并日志输出异常: {log_e}")
    return merged

def _merge_walls_iterative(walls: List[Dict[str, Any]], tol_ratio: float = 0.02, request_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """迭代执行墙段合并，直到不再产生新墙"""
    walls_list = list(walls)
    while True:
        merged_any = False
        n = len(walls_list)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_merge_wall(walls_list[i], walls_list[j], tol_ratio, request_id=request_id)
                if merged is not None:
                    walls_list = [walls_list[k] for k in range(n) if k != i and k != j]
                    walls_list.append(merged)
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return walls_list

def merge_wall_segments_in_predictions(predictions: List[Dict[str, Any]], request_id: str, tol_ratio: float = 0.02) -> List[Dict[str, Any]]:
    """在整体预测结果中合并墙段，并返回新的结果列表"""
    walls = [p for p in predictions if p.get("obj_name") == "wall" and isinstance(p.get("points"), dict)]
    others = [p for p in predictions if not (p.get("obj_name") == "wall" and isinstance(p.get("points"), dict))]
    if not walls:
        return predictions
    before = len(walls)
    merged = _merge_walls_iterative(walls, tol_ratio, request_id=request_id)
    after = len(merged)
    if after != before:
        logger.info(f"[{request_id}] 墙体合并：原墙段 {before} -> 合并后 {after}")
    return others + merged

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
                class_name_str = str(class_name).lower()
                z_value = 150.0 if class_name_str.startswith("window") else 0.0
                
                prediction = {
                    "location": {
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "z": float(z_value)
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
                class_name_str = str(class_name).lower()
                z_value = 1500.0 if class_name_str.startswith("window") else 0.0
                
                prediction = {
                    "location": {
                        "x": float(center_x),
                        "y": float(center_y),
                        "z": float(z_value)
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
        
        # 使用墙体模型进行二次预测
        try:
            if wall_model is not None:
                logger.info(f"[{request_id}] 开始使用墙体模型进行二次预测")
                wall_results = wall_model.predict(image, verbose=False)
                if wall_results:
                    wall_result = wall_results[0]
                    # 优先使用 OBB 结果
                    if hasattr(wall_result, 'obb') and wall_result.obb is not None:
                        obb_data = wall_result.obb
                        boxes = obb_data.xyxyxyxy.cpu().numpy()
                        confidences = obb_data.conf.cpu().numpy()
                        classes = obb_data.cls.cpu().numpy()
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            obb_coords = box.reshape(4, 2)
                            center, rotation = calculate_obb_center_and_rotation(obb_coords)
                            sp, ep = calculate_wall_endpoints_from_obb(obb_coords)
                            prediction = {
                                "location": {
                                    "x": float(center[0]),
                                    "y": float(center[1]),
                                    "z": 0.0
                                },
                                "points": {
                                    "start_point": {"x": float(sp[0]), "y": float(sp[1]), "z": 0.0},
                                    "end_point": {"x": float(ep[0]), "y": float(ep[1]), "z": 0.0}
                                },
                                "rotation": {"x": 0.0, "y": 0.0, "z": float(rotation)},
                                "type": 16,
                                "obj_name": "wall",
                                "confidence": float(conf)
                            }
                            predictions.append(prediction)
                            logger.info(f"[{request_id}] 墙体OBB对象 {i+1}: 置信度: {conf:.3f}, 中心: ({center[0]:.2f}, {center[1]:.2f}), 起止: ({sp[0]:.2f},{sp[1]:.2f}) -> ({ep[0]:.2f},{ep[1]:.2f}), 旋转: {rotation:.2f}°")
                    elif hasattr(wall_result, 'boxes') and wall_result.boxes is not None:
                        boxes_data = wall_result.boxes
                        boxes = boxes_data.xyxy.cpu().numpy()
                        confidences = boxes_data.conf.cpu().numpy()
                        classes = boxes_data.cls.cpu().numpy()
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            x1, y1, x2, y2 = box
                            cx = (x1 + x2) / 2.0
                            cy = (y1 + y2) / 2.0
                            w = abs(x2 - x1)
                            h = abs(y2 - y1)
                            if w <= h:
                                sp = (cx, y1)
                                ep = (cx, y2)
                            else:
                                sp = (x1, cy)
                                ep = (x2, cy)
                            prediction = {
                                "location": {
                                    "x": float(cx),
                                    "y": float(cy),
                                    "z": 0.0
                                },
                                "points": {
                                    "start_point": {"x": float(sp[0]), "y": float(sp[1]), "z": 0.0},
                                    "end_point": {"x": float(ep[0]), "y": float(ep[1]), "z": 0.0}
                                },
                                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                                "type": 16,
                                "obj_name": "wall",
                                "confidence": float(conf)
                            }
                            predictions.append(prediction)
                            logger.info(f"[{request_id}] 墙体标准框对象 {i+1}: 置信度: {conf:.3f}, 中心: ({cx:.2f}, {cy:.2f}), 起止: ({sp[0]:.2f},{sp[1]:.2f}) -> ({ep[0]:.2f},{ep[1]:.2f})")
                else:
                    logger.warning(f"[{request_id}] 墙体模型未返回任何结果")
            else:
                logger.warning(f"[{request_id}] 墙体模型未加载，跳过二次预测")
        except Exception as wall_e:
            logger.error(f"[{request_id}] 墙体模型预测失败: {wall_e}")
        # 合并墙段
        try:
            predictions = merge_wall_segments_in_predictions(predictions, request_id, tol_ratio=0.02)
        except Exception as merge_e:
            logger.error(f"[{request_id}] 墙体合并流程异常: {merge_e}")

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

    # 尝试加载墙体模型（不中断启动）
    try:
        if load_wall_model():
            logger.info("墙体模型加载成功，将进行二次预测")
        else:
            logger.warning("墙体模型加载失败，墙体相关预测将跳过")
    except Exception as e:
        logger.warning(f"加载墙体模型过程中出现异常: {e}")

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

@app.post("/predict_bytes")
async def predict_bytes(data: bytes = Body(..., description="原始二进制图片数据", media_type="application/octet-stream")):
    """
    二进制数据预测接口
    
    - 输入：请求体为二进制数据（application/octet-stream），内容为图片的二进制字节
    - 输出：与 /predict 接口一致的JSON结构
    - 处理流程：二进制数据->PIL图片(RGB)->保存->调用yolo_prediction->构建响应
    """
    start_time = time.time()

    # 生成请求ID
    timestamp = int(time.time() * 1000)
    request_id = f"req_{timestamp}"

    try:
        # 校验二进制数据
        if data is None or len(data) == 0:
            raise HTTPException(status_code=400, detail="请求体为空或不是有效的二进制图片数据")

        logger.info(f"[{request_id}] 收到二进制预测请求，数据长度: {len(data)} 字节")

        # 将二进制数据转换为图片
        try:
            image = Image.open(io.BytesIO(data))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="二进制数据无法识别为图片")

        # 转换为RGB格式（如果需要）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logger.info(f"[{request_id}] 图片加载成功，尺寸: {image.size}")

        # 保存上传的图片（与 /predict 保持一致）
        date_folder = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join("user", "image", date_folder)
        os.makedirs(save_dir, exist_ok=True)

        # 生成唯一文件名（基于内容哈希）
        file_hash = hashlib.md5(data).hexdigest()[:8]
        save_filename = f"{timestamp}_{file_hash}.jpg"
        save_path = os.path.join(save_dir, save_filename)

        # 保存图片为JPEG
        image.save(save_path, "JPEG", quality=95)
        logger.info(f"[{request_id}] 图片已保存到: {save_path}")

        # 使用真实YOLO模型进行预测
        logger.info(f"[{request_id}] 开始真实YOLO预测（二进制接口）")
        predictions = yolo_prediction(image, request_id)

        # 计算处理时间
        processing_time = time.time() - start_time

        # 构建响应（与 /predict 保持一致）
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

        logger.info(f"[{request_id}] 预测完成（二进制接口），检测到 {len(predictions)} 个对象，耗时: {processing_time:.3f}秒")

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] 二进制预测过程中发生错误: {e}")
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