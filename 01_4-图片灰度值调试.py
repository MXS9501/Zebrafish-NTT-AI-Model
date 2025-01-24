import cv2
import glob
import matplotlib.pyplot as plt

# Get list of image files
image_files = glob.glob('D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset/test-1-300/*.jpg')

# Loop through each file
for file in image_files:
    # Read the image
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    # Modify the contrast of the image
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Display the original and adjusted images using matplotlib
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Adjusted Image')
    plt.imshow(adjusted, cmap='gray')
    plt.axis('off')
    
    plt.show()
