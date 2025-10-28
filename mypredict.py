from ultralytics import YOLO

model = YOLO(r"results/yolo11n100/weights/best.pt")
result = model.predict(
    source=r"E:\deeplearning\ultralytics\predict",
    save=True,
    show=False,
)


print("result:", result)
