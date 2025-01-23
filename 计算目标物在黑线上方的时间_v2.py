import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from ultralytics import YOLO

def detect_black_dashed_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 300)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)

    black_lines = []
    if lines is not None:
        min_line_length_threshold = 500
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length > min_line_length_threshold:
                black_lines.append((x1, y1, x2, y2))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return black_lines

def is_target_above_line(target_bbox, lines):
    x_min, y_min, x_max, y_max = target_bbox
    target_center_y = (y_min + y_max) / 2

    for line in lines:
        x1, y1, x2, y2 = line
        line_center_y = (y1 + y2) / 2
        if target_center_y < line_center_y:
            return True
    return False

def main():
    model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/train15/weights/best.pt")
    folder_path = 'D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test'
    total_frames_above_line = 0
    frame_rate = 1  # Assuming a frame rate of 30 FPS

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                results = model(image)
                black_lines = detect_black_dashed_line(image)

                for result in results:
                    for bbox in result.boxes.xyxy:
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        if is_target_above_line((x_min, y_min, x_max, y_max), black_lines):
                            cv2.putText(image, "Above Line", (x_min, y_min - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                            total_frames_above_line += 1
                        else:
                            cv2.putText(image, "Below Line", (x_min, y_min - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

                cv2.imshow('Result', image)
                cv2.waitKey(10)
                cv2.destroyAllWindows()

    total_time_above_line = total_frames_above_line / frame_rate
    print(f"Total time above the black line: {total_time_above_line} seconds")
    print(f"Total position above the black line: {total_frames_above_line} pics")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

print("程序已执行完毕，将停留 10 秒钟。")
time.sleep(10)  # 程序暂停 10 秒
print("10 秒已过，程序结束。")