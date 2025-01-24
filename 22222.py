from ultralytics import YOLO
import cv2
import numpy as np
import glob

# Define the path to the YOLO model
model_path = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-backline2/weights/best.pt"

# Load a pretrained YOLO model
model = YOLO(model_path)
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
    for box in boxes:
        # Access class label
        class_label = results[0].names[box.cls[0].item()]
        if class_label == 'zebrafish':
            # Access box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0]
            # Crop the region of interest (ROI)
            roi = result_image[int(y1):int(y2), int(x1):int(x2)]
            # Convert the ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Apply thresholding to detect the black line
            _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw the contours on the original image
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

            # Detect lines in the thresholded image
            lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Display the result image
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)  # Wait indefinitely for a key press to show the next image

cv2.destroyAllWindows()
