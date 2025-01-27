from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model
model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/train16/weights/best.pt")

# Define path to the video file
source = "D:/Test_File/Test_1.mp4"  # 替换为你的视频文件路径

# Run inference on the video and display results
results = model.predict(source=source, save=True, save_txt=True)  # save=True 将检测结果保存为视频文件



# # 如果你想要实时显示检测结果，可以使用以下代码
# for result in results:
#     result_image = result.plot()  # get the image with results
#     cv2.imshow('Result', result_image)
#     cv2.waitKey(200)  # 显示下一帧，1毫秒的延迟
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 显示下一帧，1毫秒的延迟，按 'q' 键退出
#         break

# cv2.destroyAllWindows()  # 关闭所有窗口