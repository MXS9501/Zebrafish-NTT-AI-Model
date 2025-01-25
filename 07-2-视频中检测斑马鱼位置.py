from ultralytics import YOLO
import cv2
import os

# Load a pretrained YOLO model
model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-backline4/weights/best.pt")

# Define path to the video file
video_path = "D:/Test_File/Test_1.mp4"

# Define the output video file
output_video_path = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/results/result-video.mp4"

# Define the output folder for 'above' images
output_folder = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/above_images"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a counter for 'above' positions
above_count = 0

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video's frame width, height, and frames per second (fps)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Read frames from the video
while True:
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

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
            above_count += 1
            output_file = os.path.join(output_folder, f"frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
            cv2.imwrite(output_file, frame)
        elif blackline_center_y < zebrafish_center_y:
            position = 'below'
        else:
            position = 'overlap'

        # Access confidence score
        # confidence = blackline_box.conf[0]

        # Draw the position and confidence on the blackline box
        cv2.putText(frame, f"{position}", (int(zebrafish_x1), int(zebrafish_y1) - 30),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 1)

    # Write the frame with the results to the output video
    out.write(frame)

# Release the video capture object and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()

# Display the total count of 'above' positions
print(f"Total 'above' positions: {above_count}")

# Calculate the percentage of 'above' positions
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
above_percentage = (above_count / total_frames) * 100
# Display the percentage of 'above' positions
print(f"Percentage of 'above' positions: {above_percentage:.2f}%")
