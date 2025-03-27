import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLO_GUI:
    def __init__(self, master):
        self.master = master
        master.title("YOLO图像检测")

        # 加载训练好的YOLO模型（修改为你的模型路径）
        self.model = YOLO(model="network.yaml", task="detect").load("yolov8_logs/yolov8_train5/best.pt")

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 文件选择按钮
        self.btn_open = tk.Button(self.master, text="选择图片", command=self.open_image)
        self.btn_open.pack(pady=10)

        # 图像显示区域
        self.frame = tk.Frame(self.master)
        self.frame.pack()

        self.lbl_original = tk.Label(self.frame)
        self.lbl_original.pack(side=tk.LEFT, padx=10)

        self.lbl_result = tk.Label(self.frame)
        self.lbl_result.pack(side=tk.LEFT, padx=10)

    def open_image(self):
        # 打开文件对话框
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            # 进行YOLO检测
            results = self.model.predict(file_path, save=True)

            # 获取保存结果路径（根据实际路径可能需要调整）
            result_path = results[0].save_dir  # YOLOv8保存路径
            detected_img_path = f"{result_path}/{file_path.split('/')[-1]}"

            # 显示图片
            self.show_images(file_path, detected_img_path)

    def show_images(self, original_path, detected_path):
        # 打开并调整图片大小
        max_size = (600, 400)

        # 原始图片
        original_img = Image.open(original_path)
        original_img.thumbnail(max_size)
        original_tk = ImageTk.PhotoImage(original_img)
        self.lbl_original.config(image=original_tk)
        self.lbl_original.image = original_tk

        # 检测结果图片
        detected_img = Image.open(detected_path)
        detected_img.thumbnail(max_size)
        detected_tk = ImageTk.PhotoImage(detected_img)
        self.lbl_result.config(image=detected_tk)
        self.lbl_result.image = detected_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLO_GUI(root)
    root.mainloop()