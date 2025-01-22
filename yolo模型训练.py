from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from a YAML file
model = YOLO("yolo11n.pt")  # load a pretrained model from a .pt file (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML file and transfer weights from .pt file

# Train the model with verbose output to display training process
results = model.train(data="D:/Test_File/Code/Zebrafish-NTT-AI-Model/mydataset.yaml", epochs=100, batch=16)