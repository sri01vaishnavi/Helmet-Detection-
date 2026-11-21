from ultralytics import YOLO

model = YOLO(
    r"C:\Users\Public\Helmate_detection\runs\detect\runs\detect\helmet_model\weights\best.pt"
)

results = model.predict(
    source=r"C:\Users\Public\Helmate_detection\ccml\test\images",
    conf=0.5,
    save=True
)

print("Prediction completed successfully!")