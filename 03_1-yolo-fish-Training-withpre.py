from ultralytics import YOLO

# Load a model
model = YOLO("D:/Test_File/Code/Zebrafish-NTT-AI-Model/runs/detect/Fish-with-backline4 (best for now)/weights/best.pt")  # load a pretrained model

# Train the model with verbose output to display training process
results = model.train(
	data="D:/Test_File/Code/Zebrafish-NTT-AI-Model/03_2-fishdataset.yaml",  # Path to the dataset configuration file
	epochs=100,  # Number of training epochs
	batch=1,  # Batch size
	imgsz=640,  # Image size
	name="Fish-with-backline"  # Name of the model
    )

