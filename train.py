from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="ccml/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,
    device="cpu",
    project="runs/detect",
    name="helmet_model"
)