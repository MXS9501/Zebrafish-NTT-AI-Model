# 更新日志 
# 20250118 大框架拉取
# 20250119 绘制红线以评估检测效果

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def detect_black_dashed_line(image):
    # 将图像转换为 RGB 格式以便使用 matplotlib 显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_bgr = image.copy()  # 复制一份 BGR 格式的图像用于绘制红色框
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用 Canny 边缘检测算法检测图像中的边缘
    edges = cv2.Canny(gray, 200, 300)
    # 使用霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)

    if lines is not None:
        # 定义一个最小直线长度阈值，用于过滤短直线
        min_line_length_threshold = 500  # 可根据实际情况调整
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线长度
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > min_line_length_threshold:
                filtered_lines.append(line)

        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            # 绘制黑色虚线，这里使用 cv2.LINE_4 绘制虚线，你可以根据需要调整线条类型
            cv2.line(image_rgb, (x1, y1), (x2, y2), (0, 0, 225), 1, cv2.LINE_4)
            # 计算包围直线的矩形坐标
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            # 绘制红色线
            cv2.line(image_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

    # 显示原始图像
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # 显示灰度图像
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # 显示边缘图像
    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')

    # 显示带有检测到的直线的图像
    if lines is not None:
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.title('Detected Dashed Lines with Red Bounding Box')
        plt.axis('off')
    else:
        plt.subplot(2, 2, 4)
        plt.imshow(image_rgb)
        plt.title('No Dashed Lines Detected')
        plt.axis('off')

    plt.show(block=False)  # 不阻塞程序执行
    plt.pause(0.1)
    time.sleep(2)  # 显示 0.5 秒
    plt.close()  # 关闭图像显示窗口


def main():
    folder_path = 'D:\\Test_File\\Line_Test'  # 请将此路径替换为你自己的文件夹路径
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 可以添加更多图像文件的后缀
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                detect_black_dashed_line(image)


if __name__ == "__main__":
    main()