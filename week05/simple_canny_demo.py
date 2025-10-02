import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image

print("=== Canny(image, 100, 200) 详细解析 ===")

# 下载并处理图像
image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
image_array = np.array(image)
gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

print(f"图像尺寸: {gray.shape}")

# 测试不同参数
params = [
    (50, 100),   # 低阈值
    (100, 200),  # 原始参数
    (150, 300),  # 高阈值
    (30, 60),    # 很低阈值
    (200, 400)   # 很高阈值
]

for low, high in params:
    edges = cv2.Canny(gray, low, high)
    edge_count = np.sum(edges > 0)
    density = edge_count / edges.size * 100
    
    print(f"\nCanny({low}, {high}):")
    print(f"  边缘像素: {edge_count:,}")
    print(f"  密度: {density:.2f}%")
    
    # 保存图像
    edge_img = Image.fromarray(edges)
    edge_img.save(f"edges_{low}_{high}.png")
    print(f"  已保存: edges_{low}_{high}.png")

print(f"\n=== 参数含义 ===")
print(f"cv2.Canny(image, threshold1, threshold2)")
print(f"- threshold1 (100): 低阈值，决定弱边缘")
print(f"- threshold2 (200): 高阈值，决定强边缘")
print(f"- 算法会先找到所有梯度>200的强边缘")
print(f"- 然后连接与强边缘相邻且梯度>100的弱边缘")
print(f"- 孤立的弱边缘会被丢弃")

print(f"\n参数修改效果:")
print(f"- 降低阈值 → 更多边缘和细节，但可能有噪声")
print(f"- 提高阈值 → 只保留最强边缘，结果更简洁")
print(f"- 高低阈值比例建议保持在2:1到3:1之间")
