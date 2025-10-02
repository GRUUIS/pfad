import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

print("=== Canny边缘检测详细解析 ===\n")

# 下载测试图像
print("正在下载测试图像...")
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# 转换为灰度图像（Canny需要灰度图）
if len(image.shape) == 3:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray = image

print(f"图像尺寸: {gray.shape}")
print(f"总像素数: {gray.size}")

# 测试不同的Canny参数
test_cases = [
    (50, 100, "低阈值设置"),
    (100, 200, "标准设置（原代码）"),
    (150, 300, "高阈值设置"),
    (30, 60, "很低阈值（包含噪声）"),
    (200, 400, "很高阈值（只有强边缘）"),
    (100, 120, "接近的双阈值"),
    (50, 300, "阈值差距大")
]

print("\n=== 不同参数测试结果 ===")
results = []

for low, high, desc in test_cases:
    # 应用Canny边缘检测
    edges = cv2.Canny(gray, low, high)
    
    # 统计边缘像素
    edge_pixels = np.sum(edges > 0)
    edge_density = edge_pixels / edges.size * 100
    
    results.append({
        'low': low,
        'high': high,
        'desc': desc,
        'edge_pixels': edge_pixels,
        'edge_density': edge_density,
        'edges': edges
    })
    
    print(f"\n参数组合: ({low}, {high}) - {desc}")
    print(f"  边缘像素数: {edge_pixels:,}")
    print(f"  边缘密度: {edge_density:.2f}%")
    print(f"  阈值比例: {high/low:.1f}:1")

# 保存一些示例图像
print(f"\n正在保存示例图像...")
for i, result in enumerate(results[:4]):  # 保存前4个示例
    # 转换为3通道以便显示
    edges_3ch = np.stack([result['edges']] * 3, axis=2)
    edge_image = Image.fromarray(edges_3ch)
    filename = f"canny_example_{result['low']}_{result['high']}.png"
    edge_image.save(filename)
    print(f"  已保存: {filename}")

print(f"\n=== Canny算法工作原理详解 ===")
print(f"""
1. 高斯滤波降噪:
   - 使用高斯核平滑图像，减少噪声影响
   
2. 计算梯度:
   - 使用Sobel算子计算x和y方向的梯度
   - 梯度幅值 = sqrt(Gx² + Gy²)
   - 梯度方向 = arctan(Gy/Gx)
   
3. 非最大值抑制:
   - 沿梯度方向，只保留局部最大值
   - 使边缘变细，通常为1像素宽
   
4. 双阈值检测:
   - 高阈值({test_cases[1][1]}): 梯度 > {test_cases[1][1]} → 强边缘(确定是边缘)
   - 低阈值({test_cases[1][0]}): 梯度 < {test_cases[1][0]} → 非边缘(确定不是边缘)  
   - 中间值: {test_cases[1][0]} < 梯度 < {test_cases[1][1]} → 弱边缘(可能是边缘)
   
5. 边缘连接 (滞后阈值):
   - 弱边缘像素只有连接到强边缘时才被保留
   - 孤立的弱边缘被丢弃
""")

print(f"\n=== 参数调整建议 ===")
print(f"""
调低阈值的效果:
✓ 检测更多细节和纹理
✓ 对微弱边缘敏感
✗ 可能包含噪声
✗ 边缘可能不够清晰

调高阈值的效果:
✓ 只保留最明显的边缘
✓ 结果更清晰简洁
✓ 抗噪声能力强
✗ 可能丢失重要的弱边缘
✗ 细节信息减少

阈值比例建议:
- 标准比例: 高:低 = 2:1 到 3:1
- 如 (100,200), (150,300), (50,150)
- 比例太小(<1.5): 很少弱边缘被连接
- 比例太大(>4): 可能连接噪声

实际应用场景:
- ControlNet图像生成: (100,200) - 平衡细节和清晰度
- 建筑轮廓提取: (150,300) - 只要主要结构
- 纹理分析: (50,100) - 保留更多细节
- 噪声图像: (120,240) - 提高阈值抗噪声
""")

print(f"\n演示完成！请查看生成的示例图像文件。")
