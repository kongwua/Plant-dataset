import torch
from ultralytics import YOLO
from torch.optim.lr_scheduler import CosineAnnealingLR
# from modules import *
# from ultralytics.nn.tasks import parse_model

# # 注册自定义模块
# globals()["CARAFE"] = CARAFE
# globals()["Mlt_ECA"] = Mlt_ECA

model = YOLO("network.yaml").load("yolov8_logs/yolov8_train3/weights/last.pt")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)
# model = YOLO("network.yaml")
# model = YOLO("yolov8n.pt")
# model.lr_find(data='dataset.yaml')

# 训练模型
def main():
    results = model.train(
        data='dataset.yaml',
        epochs=1000,  # 50 100 300
        imgsz=320,
        workers=2,
        batch=8,
        device='cuda',

        
        project='yolov8_logs',
        name='yolov8_train3',
        save_period=100,  # 保存模型的周期
        lr0=0.01,  # 初始学习率
        lrf=0.001,  # 最终学习率
        exist_ok=True,  #
        pretrained=True,
        amp=True,  #
        val=True,  #
        iou=0.5,
        plots=True  #
    )

if __name__ == '__main__':
    main()