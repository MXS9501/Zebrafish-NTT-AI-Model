import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("D:/Zebrafish Project/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-blackline_1/weights/best.pt")

# Open the video file
video_path = "D:/Zebrafish Project/Test_1.mp4"
cap = cv2.VideoCapture(video_path)

# 初始化一个字典来存储每个目标的轨迹
tracks = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # 遍历每个检测结果
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    # 计算目标的中心点
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 如果该目标的轨迹还未记录，则初始化一个列表
                    if track_id not in tracks:
                        tracks[track_id] = []

                    # 添加当前中心点到轨迹列表
                    tracks[track_id].append((center_x, center_y))

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # 绘制轨迹
        for track_id in tracks:
            if len(tracks[track_id]) > 1:
                for i in range(1, len(tracks[track_id])):
                    cv2.line(annotated_frame, tracks[track_id][i - 1], tracks[track_id][i], (0, 255, 0), 2)

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()