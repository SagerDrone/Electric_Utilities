from ultralytics import YOLO

# Paths to the config file and output directory
config_path = r"config.yaml"
output_dir = r"C:\Users\Sager\Desktop\Sager\OHTL Data InsPLAD\InsPLAD-det\output"

# Initialize the YOLO model with the YOLOv8n architecture
model = YOLO(r"C:\Users\Sager\Desktop\Sager\OHTL Data InsPLAD\InsPLAD-det\yolov8n.pt")  # Pre-trained YOLOv8n weights

# Train the model
print("Starting training...")
model.train(data=config_path, epochs=50, batch=16, imgsz=640, project=output_dir, name="yolov8n_model")
print("Training complete.")

# Validate the model
print("Starting validation...")
results = model.val(data=config_path, project=output_dir, name="yolov8n_model_validation")
print("Validation complete.")

# Print validation metrics
print("Validation Results:")
print(results)
