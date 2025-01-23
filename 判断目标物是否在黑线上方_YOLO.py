from ultralytics import YOLO
import cv2
import numpy as np
import os

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
    model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/train4/weights/best.pt")
    folder_path = 'D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test-1-300'
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                results = model(image)
                black_lines = detect_black_dashed_line(image)

                for result in results:
                    for bbox, conf in zip(result.boxes.xyxy, result.boxes.conf):
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        class_name = result.names[int(result.boxes.cls[0])]
                        label = f"{class_name} P: {conf:.2f}"
                        if is_target_above_line((x_min, y_min, x_max, y_max), black_lines):
                            label += " Above"
                        else:
                            label += " Below"
                        cv2.putText(image, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                        
                        output_folder_above = 'D:/Test_File/Code/Zebrafish-NTT-AI-Model/output/above'
                        output_folder_below = 'D:/Test_File/Code/Zebrafish-NTT-AI-Model/output/below'
                        
                        if not os.path.exists(output_folder_above):
                            os.makedirs(output_folder_above)
                        if not os.path.exists(output_folder_below):
                            os.makedirs(output_folder_below)
                        
                        if "Above" in label:
                            output_path = os.path.join(output_folder_above, filename)
                        else:
                            output_path = os.path.join(output_folder_below, filename)
                        
                        cv2.imwrite(output_path, image)
                        
                        # # 使用 cv2 显示结果图像
                        # cv2.imshow('Detection Results', image)
                        # cv2.waitKey(200)  # 等待按键
                        cv2.destroyAllWindows()  # 关闭所有窗口
                     

if __name__ == "__main__":
    main()