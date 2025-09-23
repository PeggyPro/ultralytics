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

torch.backends.cudnn.benchmark = True  # 固定尺寸更快

if __name__ == '__main__':
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    print("ultralytics.__version__:", ultralytics.__version__)

    # 4GB 显存建议使用 nano 模型
    a1 = YOLO('yolo11m.pt')  # 若显存不够再退回 yolo11s.pt

    a1.train(
        data='D:/Users/fh704/D-Documents/D-Github/ultralytics/studyYolo/data_stjdb.yaml',
        epochs=200,                  # 配合早停，等价于 150~220 的有效训练
        imgsz=960,                   # 832/960 二选一；显存足够就 960
        batch=2,                    # 先用 24，观察显存后再试 28/32，直到接近上限
        device=0,
        workers=4,                   # 依 CPU 再试 8/12/16，观察 CPU 占用
        amp=False,                   # 1080（Pascal）建议关 AMP，换显存来提吞吐
        cache=True,                  # 内存≥32GB 就开；不足则 False
        val=True,
        patience=50,                 # 早停，稳定后自动停止

        # 学习率与调度（保持稳健）
        lr0=0.01,                    # YOLO 的 nbs 缩放会自动按 batch 比例调整
        warmup_epochs=3,
        nbs=64,
        cos_lr=True,
        close_mosaic=10,

        # 数据增强：减轻 CPU 端重负，避免“喂不动”
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, fliplr=0.5,

        # 其他
        project='runs/train',
        name='y11m_960_b24_balanced'
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
