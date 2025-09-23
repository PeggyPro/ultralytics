# 开始训练模型
# 调参:  https://blog.csdn.net/blink182007/article/details/136928129

# x06开始训练模型.py

############################## 1.这是分批次训练 #################################
import torch
from ultralytics import YOLO
import ultralytics
import matplotlib.pyplot as plt
from matplotlib import font_manager

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 确保中文字体可以被matplotlib识别
font_path = "C:/Windows/Fonts/simsun.ttc"  # 替换为你系统中实际的中文字体路径
font_prop = font_manager.FontProperties(fname=font_path)

# 使用该字体设置matplotlib的全局参数
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

if __name__ == '__main__':
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    print("ultralytics.__version__:", ultralytics.__version__)

    # 4GB 显存建议使用 nano 模型
    a1 = YOLO('yolo11s.pt')

    a1.train(
        data='D:/Users/fh704/D-Documents/D-Github/ultralytics/studyYolo/data_stjdb.yaml',
        epochs=200,          # 6500 张样本建议 120~200；配合早停
        imgsz=640,           # 4GB 显存更稳的分辨率；小目标多再考虑 640
        batch=-1,            # AutoBatch 自动探测最大可用 batch
        device=0,
        workers=4,           # 增加workers充分利用CPU（如果CPU性能足够）
        amp=True,            # 混合精度，省显存+提速
        val=True,
        patience=50,         # 早停
        cache=False,         # 使用RAM缓存加速数据加载（需要≥32GB内存）
        seed=0,

        # 最大化性能的参数设置
        lr0=0.01,            # 恢复默认学习率，充分利用大batch
        warmup_epochs=3,     # 减少warmup，更快进入正常训练
        nbs=64,              # 保持默认nominal batch size
        cos_lr=True,         # 使用余弦学习率调度，训练更稳定
        close_mosaic=10,     # 最后10个epoch关闭mosaic增强，提高精度

        # 数据增强增强（充分利用显存）
        hsv_h=0.015,         # 色调增强
        hsv_s=0.7,           # 饱和度增强
        hsv_v=0.4,           # 亮度增强
        translate=0.1,       # 平移增强
        scale=0.5,           # 缩放增强
        fliplr=0.5,          # 水平翻转
        mosaic=1.0,          # mosaic数据增强

        project='runs/train',
        name='y11s_640_e1024_8GB_max'
    )

    print('训练完成')



############################## 2.这是一次训练所有的 #################################

# if __name__ == '__main__':
#     # 检测GPU是否可用
#     print("torch.__version__: ", torch.__version__)
#     print("torch.version.cuda: ", torch.version.cuda)
#     print("torch.cuda.is_available: ", torch.cuda.is_available())
#     print("torch.cuda.device_count: ", torch.cuda.device_count())
#
#     flag = torch.cuda.is_available()
#     if flag:
#         print("CUDA可使用")
#     else:
#         print("CUDA不可用")
#
#     # 加载预训练模型
#     a1 = YOLO('yolo11n.pt')
#     # 开始训练模型
#     a1.train(
#         data='./data_stjdb.yaml',  # 数据集配置文件路径
#         epochs=500,  # 训练轮数
#         imgsz=640,  # 输入图片尺寸, 官方推荐640
#         batch=16,  # 批处理大小, 官方推荐16/32
#         device='0',  # GPU=0 或者 CPU='cpu'
#     )
#
#     print('训练完成')
