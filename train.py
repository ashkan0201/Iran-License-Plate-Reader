from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,  
    degrees=10,
    translate=0.08,
    scale=0.2,
    shear=2,
    perspective=0.0003,
    fliplr=0.5,
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
)
