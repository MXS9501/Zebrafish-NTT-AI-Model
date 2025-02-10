from seaborn import lmplot
from ultralytics import YOLO
import cv2
import glob
import os
from alive_progress import alive_bar
import time
start_time = time.time()

# Load a pretrained YOLO model
model = YOLO("D:/Zebrafish Project/Code/Zebrafish-NTT-AI-Model/runs/detect/train16/weights/best.pt")
# Define path to the image files
# source = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test-1-300/*.jpg"
source = "D:/Zebrafish Project/Code/Zebrafish-NTT-AI-Model/mydataset/test-1-300/*.jpg"

##############################################################################
# Define the output folder for 'above' images
# output_folder = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/above_images"
# Create the output folder if it doesn't exist
# if not os.path.exists(output_folder):
    # os.makedirs(output_folder)
##############################################################################
# Get list of image files
image_files = glob.glob(source)
##############################################################################
# Initialize a counter for 'above' positions
above_count = 0
##############################################################################
# #Set the confidence threshold
# confidence_threshold = 0.9
##############################################################################
# Initialize a progress bar
with alive_bar(len(image_files)) as bar:
##############################################################################
# Run inference on each image and display results
    for image_file in image_files:
        results = model.predict(image_file)  # list of Results objects
        result_image = results[0].plot()  # get the image with results
##############################################################################        
        # boxes = results[0].boxes[results[0].boxes.conf > confidence_threshold]
##############################################################################
        # Get bounding boxes from the results
        boxes = results[0].boxes
        zebrafish_box = None
        blackline_box = None
        for box in boxes:
            # Access class label
            class_label = results[0].names[box.cls[0].item()]
            if class_label == 'zebrafish':
                zebrafish_box = box
            elif class_label == 'blackline':
                blackline_box = box

        if zebrafish_box is not None and blackline_box is not None:
            # Access box coordinates (xyxy format)
            zebrafish_x1, zebrafish_y1, zebrafish_x2, zebrafish_y2 = zebrafish_box.xyxy[0]
            blackline_x1, blackline_y1, blackline_x2, blackline_y2 = blackline_box.xyxy[0]

            # Calculate the center points
            zebrafish_center_y = (zebrafish_y1 + zebrafish_y2) / 2
            blackline_center_y = (blackline_y1 + blackline_y2) / 2

            # Compare the center points
            if blackline_center_y > zebrafish_center_y:
                position = 'above'
##############################################################################
                above_count += 1
##############################################################################
                # output_file = os.path.join(output_folder, os.path.basename(image_file))
                # cv2.imwrite(output_file, result_image)
##############################################################################
            elif blackline_center_y < zebrafish_center_y:
                position = 'below'
            else:
                position = 'overlap'

            # Access confidence score
            # confidence = blackline_box.conf[0]

            # Draw the position and confidence on the blackline box
            if position == 'above':
                color = (0, 255, 0)  # Green
            elif position == 'below':
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue for overlap

            cv2.putText(result_image, f"{position}", (int(zebrafish_x1), int(zebrafish_y1) - 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1)
##############################################################################  
        bar()
##############################################################################
        # Display the result image
        cv2.imshow('Result', result_image)
        cv2.waitKey(1)  # Wait for a key press to show the next image
    # cv2.destroyAllWindows()
##############################################################################
from PyQt5.QtWidgets import QApplication, QMessageBox


if above_count > 0:
    app = QApplication([])
    msg_box = QMessageBox()
    msg_box.setWindowTitle("Counting Result")
    msg_box.setText(f"Total [-{above_count}-] Frames are above")
    msg_box.exec_()
    # app.exec_()
#############################################################################

# 记录程序结束时间
end_time = time.time()

# 计算总用时
total_time = end_time - start_time

# 显示总用时
print(f"程序运行总用时: {total_time} 秒")


