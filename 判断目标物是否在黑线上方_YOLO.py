# 更新日志 
# 20250118 大框架拉取
# 20250119 
# python
# ===================== #
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


# 图像预处理（用于目标检测）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 假设使用 YOLOv5 进行目标检测（需要先安装 YOLOv5）
# 可以使用以下命令安装 YOLOv5：pip install yolov5
from yolov5 import detect

def detect_target_a(image_path):
    # 使用 YOLOv5 进行目标检测
    results = detect.run(source=image_path, weights='yolov5s.pt')
    # 这里假设结果中包含目标 A 的信息，需要根据实际情况调整
    # 以下是一个简单的假设，实际情况可能需要从 results 中提取更准确的信息
    target_a_detected = False
    for result in results.xyxy[0]:
        if result[-1] == 'target_a':  # 假设目标 A 的类别名为 target_a
            target_a_detected = True
    return target_a_detected


def detect_black_line(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    if lines is not None:
        # 这里假设只存在一条黑线，取第一条检测到的黑线
        x1, y1, x2, y2 = lines[0][0]
        return (x1, y1, x2, y2)
    else:
        return None


def is_target_a_above_black_line(image_path):
    black_line = detect_black_line(image_path)
    target_a_detected = detect_target_a(image_path)
    if black_line and target_a_detected:
        x1, y1, x2, y2 = black_line
        # 假设目标 A 的位置信息通过目标检测得到，这里使用简单的假设位置 (x, y, w, h)
        target_a_position = (100, 50, 50, 50)  # 需根据实际情况调整
        target_a_y_center = target_a_position[1] + target_a_position[3] / 2
        # 判断目标 A 的中心是否在黑线上方
        if target_a_y_center < y1:
            return True
    return False


def main():
    image_dir = 'path/to/your/image/frames'  # 存储帧图像的目录
    frame_rate = 30  # 假设视频帧率为 30 帧/秒
    total_frames = 0
    target_a_above_black_line_frames = 0
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        if is_target_a_above_black_line(image_path):
            target_a_above_black_line_frames += 1
        total_frames += 1
    # 计算累计时长（秒）
    accumulated_duration = target_a_above_black_line_frames / frame_rate
    print(f"Target A is above the black line for {accumulated_duration} seconds.")


if __name__ == "__main__":
    main()
# 代码解释：
# 图像预处理（用于目标检测）：
# transforms.Compose：将图像转换为适合目标检测模型的输入格式。
# 目标 A 检测：
# detect_target_a：使用 YOLOv5 进行目标检测，这里假设 YOLOv5 已经安装并可以正常使用。
# detect.run：调用 YOLOv5 进行目标检测，weights='yolov5s.pt' 表示使用预训练的 YOLOv5 小模型，可根据需要替换为其他权重文件。
# 黑线检测：
# detect_black_line：使用 Canny 边缘检测和霍夫变换检测黑线，假设视频中只有一条黑线。
# 判断目标 A 是否在黑线上方：
# is_target_a_above_black_line：结合黑线检测和目标 A 检测结果，判断目标 A 是否在黑线上方。
# 这里假设目标 A 的位置是简单的矩形区域 (x, y, w, h)，需要根据实际的目标检测结果调整。
# 主函数：
# main：遍历存储帧图像的目录，调用 is_target_a_above_black_line 函数判断目标 A 是否在黑线上方，并统计帧数。
# 根据帧率计算累计时长。
# 使用说明：
# 请将 image_dir 替换为存储帧图像的实际目录。
# 确保已经正确安装 YOLOv5 并下载了相应的预训练权重文件。
# 根据实际情况调整目标 A 的位置信息提取和判断逻辑。
# 注意事项：
# 上述代码中关于目标 A 位置的提取是一个简单假设，实际应用中需要根据 YOLOv5 的检测结果进行详细调整。
# 对于黑线的检测，这里仅使用了简单的 OpenCV 方法，如果黑线检测效果不佳，可以考虑使用更复杂的图像处理或深度学习方法。
# 视频帧率需要根据实际情况进行调整。