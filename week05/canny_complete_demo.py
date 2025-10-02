"""
Canny边缘检测详细演示 - cv2.Canny(image, 100, 200) 参数解析
"""
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

print("=== Canny边缘检测参数详解 ===\n")

# 下载测试图像
print("下载测试图像...")
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
original_array = np.array(image)
print(f"原始图像尺寸: {original_array.shape}")

# 保存原始图像用于对比
Image.fromarray(original_array).save("0_original.png")
print("已保存: 0_original.png")

print(f"\n=== cv2.Canny(image, 100, 200) 详细分析 ===")

# 测试不同的Canny参数组合
test_cases = [
    # (low_threshold, high_threshold, 描述, 预期效果)
    (25, 50, "极低阈值", "最多边缘，包含大量噪声和纹理"),
    (50, 100, "低阈值", "较多边缘和细节"),
    (100, 200, "标准阈值(原代码)", "平衡的边缘检测效果"),
    (150, 300, "高阈值", "只保留强边缘"),
    (200, 400, "极高阈值", "只有最明显的边缘"),
    (100, 120, "接近双阈值", "很少弱边缘被连接"),
    (50, 300, "大阈值差", "更多弱边缘被保留")
]

results = []

for i, (low, high, name, description) in enumerate(test_cases):
    print(f"\n{i+1}. 测试参数: cv2.Canny(image, {low}, {high}) - {name}")
    print(f"   预期效果: {description}")
    
    # 应用Canny边缘检测
    edges = cv2.Canny(original_array, low, high)
    
    # 统计信息
    total_pixels = edges.size
    edge_pixels = np.sum(edges > 0)
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    print(f"   结果统计:")
    print(f"   - 总像素数: {total_pixels:,}")
    print(f"   - 边缘像素数: {edge_pixels:,}")
    print(f"   - 边缘占比: {edge_percentage:.2f}%")
    print(f"   - 阈值比例: {high/low:.1f}:1")
    
    # 转换为3通道图像以便显示和保存
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    
    # 保存结果
    filename = f"{i+1}_canny_{low}_{high}_{name.replace(' ', '_')}.png"
    Image.fromarray(edges_3ch).save(filename)
    print(f"   已保存: {filename}")
    
    results.append({
        'params': (low, high),
        'name': name,
        'edge_pixels': edge_pixels,
        'percentage': edge_percentage,
        'edges': edges
    })

print(f"\n=== Canny算法工作流程详解 ===")
print(f"""
步骤1: 高斯滤波
- 使用5x5高斯核对图像进行平滑处理
- 目的: 减少噪声对边缘检测的影响

步骤2: 计算梯度
- 使用Sobel算子计算x和y方向的梯度
- 梯度幅值: sqrt(Gx² + Gy²)
- 梯度方向: arctan(Gy/Gx)

步骤3: 非最大值抑制
- 沿着梯度方向，只保留局部最大值
- 目的: 将粗边缘细化为1像素宽的细边缘

步骤4: 双阈值检测
- 高阈值({test_cases[2][1]}): 梯度幅值 > {test_cases[2][1]} → 强边缘(确定保留)
- 低阈值({test_cases[2][0]}): 梯度幅值 < {test_cases[2][0]} → 非边缘(确定丢弃)
- 中间区域: {test_cases[2][0]} < 梯度幅值 < {test_cases[2][1]} → 弱边缘(待定)

步骤5: 边缘连接(滞后阈值)
- 弱边缘像素只有连接到强边缘时才被保留
- 孤立的弱边缘像素被丢弃
- 这样可以保持边缘连续性，同时抑制噪声
""")

print(f"\n=== 参数调整指南 ===")
print(f"""
低阈值(threshold1)的影响:
- 降低 → 检测更多细节，但可能引入噪声
- 提高 → 减少噪声，但可能丢失细节

高阈值(threshold2)的影响:  
- 降低 → 更多边缘被认定为强边缘
- 提高 → 只有最明显的边缘被保留

阈值比例建议:
- 理想比例: 高阈值 = 2~3 × 低阈值
- 比例过小: 很少弱边缘被连接，边缘不连续
- 比例过大: 可能连接噪声，边缘质量下降

具体应用建议:
- ControlNet(AI图像生成): (100, 200) - 平衡细节和清晰度
- 工业检测: (150, 300) - 只要清晰边缘，避免噪声
- 艺术创作: (50, 150) - 保留更多纹理和细节
- 医学图像: (80, 160) - 谨慎平衡，避免丢失重要信息
""")

# 对比分析
print(f"\n=== 结果对比分析 ===")
results.sort(key=lambda x: x['percentage'])
print("从边缘密度排序 (从少到多):")
for result in results:
    low, high = result['params']
    print(f"  {result['name']:15} ({low:3},{high:3}): {result['percentage']:5.2f}% 边缘密度")

print(f"\n演示完成! 请查看生成的图像文件来对比不同参数的视觉效果。")
