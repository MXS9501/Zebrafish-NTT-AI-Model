# 更新日志
import cv2
def split_video_into_frames(video_path, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 如果读取失败，说明已经到视频末尾，退出循环
        if not ret:
            break
        # 生成输出文件名，使用零填充确保文件名数字位数一致
        output_filename = f"{output_path}/frame_{frame_count:04d}.jpg"
        # 保存当前帧为图像文件
        cv2.imwrite(output_filename, frame)
        frame_count += 1
    # 释放视频文件资源
    cap.release()
    print(f"Total frames: {frame_count}")


# 示例使用
video_path = 'D:/Test_File/Test_1.mp4'  # 输入视频文件的路径
output_path = 'D:/Test_File/Output'  # 输出帧的存储路径
split_video_into_frames(video_path, output_path)

import time

print("程序已执行完毕，将停留 10 秒钟。")
time.sleep(10)  # 程序暂停 10 秒
print("10 秒已过，程序结束。")