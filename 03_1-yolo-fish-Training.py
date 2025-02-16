from ultralytics import YOLO

def main():
    # Load a new model
    model = YOLO("yolo11n.yaml")  # build a new model from a YAML file
    model = YOLO("yolo11n.pt")  # load a pretrained model from a .pt file (quick and easy, recommended for training)
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML file and transfer weights from .pt file (allows customization of model architecture)
    
    # Train the model with verbose output to display training process
    results = model.train(
        data="D:/Zebrafish Project/Code/Zebrafish-NTT-AI-Model/03_2-fishdataset.yaml",  # Path to the dataset configuration file
        epochs=50,  # Number of training epochs
        batch=1,  # Batch size
        imgsz=640,  # Image size
        name="Fish-with-blackline_1", # Name of the model
        device=0,  # Index of the GPU to use (0 for the first GPU)
    )

if __name__ == '__main__':
    main()
############################################################################################################
# @software{yolo11_ultralytics,
#   author = {Glenn Jocher and Jing Qiu},
#   title = {Ultralytics YOLO11},
#   version = {11.0.0},
#   year = {2024},
#   url = {https://github.com/ultralytics/ultralytics},
#   orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
#   license = {AGPL-3.0}
############################################################################################################
