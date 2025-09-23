# 加载预训练模型
# 安装第三方库: ultralytics
# 导包: from ultralytics import YOLO
from ultralytics import YOLO

# 1. 加载模型
# 2. 检测目标
# 加载预训练模型
# yolov8n.pt是官方提供的基础测试和训练模型
# 首次运行自动下载
a1 = YOLO('yolo11n.pt')
# 2. 检测目标
# show=True 显示检测结果
# save=True 保存检测结果
a1('1.png', show=True, save=True)
