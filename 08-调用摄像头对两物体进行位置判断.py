import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import os

# 加载YOLO模型
model = YOLO("D:/Zebrafish Project/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-blackline_1/weights/best.pt")

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率
width = 640  # 设置宽度
height = 360  # 设置高度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# 设置输出文件夹
output_folder = 'D:/Zebrafish Project/Recordings/'
# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 设置输出路径
output_path = f'{output_folder}recording_{timestamp}.avi'

# 定义视频编码器和创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

# 初始化变量
total_time = 0
start_time = None
fps = cap.get(cv2.CAP_PROP_FPS)
confidence_threshold = 0.5  # 设置置信度阈值
recording_start_time = time.time()
recording_duration = 1 * 60  # 5 分钟

while cap.isOpened() and (time.time() - recording_start_time) < recording_duration:
    # 读取一帧图像
    success, frame = cap.read()

    if success:
        # 使用YOLO模型进行目标检测
        results = model(frame)

        # 可视化检测结果
        annotated_frame = results[0].plot()

        # 写入帧到视频文件
        out.write(annotated_frame)

        zebrafish_above_blackline = False
        zebrafish_centers = []
        blackline_centers = []

        # 遍历检测结果，找出zebrafish和blackline的中心点
        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            confidence = box.conf[0].cpu().numpy()  # 获取置信度
            if class_name == 'zebrafish' and confidence >= confidence_threshold:
                zebrafish_centers.append((center_x, center_y))
            elif class_name == 'blackline':
                blackline_centers.append((center_x, center_y))

        # 检查zebrafish中心点是否在blackline中心点上方
        for zebrafish_center in zebrafish_centers:
            for blackline_center in blackline_centers:
                if zebrafish_center[1] < blackline_center[1]:
                    zebrafish_above_blackline = True
                    break
            if zebrafish_above_blackline:
                break

        # 计算时长
        if zebrafish_above_blackline:
            if start_time is None:
                start_time = time.time()
        else:
            if start_time is not None:
                end_time = time.time()
                total_time += end_time - start_time
                start_time = None

        # 显示结果
        cv2.imshow("Live Inference", annotated_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # 无法读取帧，退出循环
        break

# 处理最后一段时间
if start_time is not None:
    end_time = time.time()
    total_time += end_time - start_time

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"zebrafish位于blackline上方的总时长: {total_time} 秒")

##############################################################################
from PyQt5.QtWidgets import QApplication, QMessageBox


if total_time > 0:
    app = QApplication([])
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Counting Result")
    msg_box.setText(f"zebrafish位于blackline上方的总时长: {total_time} 秒")
    msg_box.exec_()
    # app.exec_()
#############################################################################

