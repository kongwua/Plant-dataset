from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("network.yaml").load("yolov8_logs/yolov8_train4/best.pt")

# 运行验证
def main():
    results = model.val(
        data="dataset.yaml",
        batch=8,
        imgsz=640,
        device="cuda",  # 可选 'cpu' 或 GPU 索引（如 '0'）
        split="val",  # 验证集划分
        conf=0.5,     # 置信度阈值
        iou=0.6,      # IoU 阈值
    )

if __name__ == '__main__':
    main()