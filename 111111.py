from ultralytics import YOLO
import cv2
import glob

# Load a pretrained YOLO model
model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-backline2/weights/best.pt")
# Define path to the image files
source = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test-1-300/*.jpg"

# Get list of image files
image_files = glob.glob(source)

# Run inference on each image and display results
for image_file in image_files:
    results = model(image_file)  # list of Results objects
    result_image = results[0].plot()  # get the image with results

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
        if blackline_center_y < zebrafish_center_y:
            position = 'above'
        else:
            position = 'below'

        # Access confidence score
        confidence = blackline_box.conf[0]

        # Draw the position and confidence on the blackline box
        cv2.putText(result_image, f"{position} ({confidence:.2f})", (int(blackline_x1), int(blackline_y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the result image
    cv2.imshow('Result', result_image)
    cv2.waitKey()  # Wait for a key press to show the next image

cv2.destroyAllWindows()
