"""
API测试脚本
用于测试YOLO OBB预测API的功能
"""

import requests
import json
from pathlib import Path

def test_api_health():
    """测试API健康检查"""
    try:
        response = requests.get("http://localhost:8000/health")
        print("=== 健康检查测试 ===")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_api_root():
    """测试API根路径"""
    try:
        response = requests.get("http://localhost:8000/")
        print("\n=== 根路径测试 ===")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"根路径测试失败: {e}")
        return False

def test_predict_api():
    """测试预测API"""
    try:
        # 查找测试图片
        predict_dir = Path("predict")
        if not predict_dir.exists():
            print("predict目录不存在，无法进行预测测试")
            return False
        
        # 获取第一张图片
        image_files = list(predict_dir.glob("*.jpg"))
        if not image_files:
            print("predict目录中没有找到jpg图片文件")
            return False
        
        test_image = image_files[0]
        print(f"\n=== 预测API测试 ===")
        print(f"使用测试图片: {test_image}")
        
        # 发送预测请求
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.name, f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/predict", files=files)
        
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 验证响应格式
            if "status" in result and "results" in result:
                print("\n=== 响应格式验证 ===")
                print(f"状态: {result['status']}")
                print(f"检测对象数量: {result.get('total_objects', 0)}")
                
                # 验证每个检测结果的格式
                for i, obj in enumerate(result.get('results', [])):
                    print(f"\n对象 {i+1}:")
                    print(f"  位置: {obj.get('location', {})}")
                    print(f"  旋转: {obj.get('rotation', {})}")
                    print(f"  类型: {obj.get('type', 'N/A')}")
                    print(f"  描述: {obj.get('description', 'N/A')}")
                    print(f"  置信度: {obj.get('confidence', 'N/A')}")
                
                return True
            else:
                print("响应格式不正确")
                return False
        else:
            print(f"预测失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"预测API测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试YOLO OBB预测API...")
    
    tests = [
        ("API健康检查", test_api_health),
        ("API根路径", test_api_root),
        ("预测API", test_predict_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
        print(f"{test_name}: {'通过' if result else '失败'}")
    
    print(f"\n{'='*50}")
    print("测试总结:")
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n总计: {passed}/{total} 个测试通过")

if __name__ == "__main__":
    main()