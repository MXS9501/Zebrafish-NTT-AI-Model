from ultralytics import YOLO
import cv2
import glob

# Load a pretrained YOLO model
model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/train4/weights/best.pt")
# Define path to the image files
source = "D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test-3-300/*.jpg"

# Get list of image files
image_files = glob.glob(source)

# Run inference on each image and display results
for image_file in image_files:
    results = model(image_file)  # list of Results objects
    result_image = results[0].plot()  # get the image with results

    # Display the result image
    cv2.imshow('Result', result_image)
    cv2.waitKey(400)  # Wait for a key press to show the next image

cv2.destroyAllWindows()
