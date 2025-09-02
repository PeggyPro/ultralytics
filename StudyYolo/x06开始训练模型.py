# 开始训练模型
# 调参:  https://blog.csdn.net/blink182007/article/details/136928129

# x06开始训练模型.py

############################## 这是分批次训练 #################################
import torch

from ultralytics import YOLO

if __name__ == '__main__':
    print("torch.__version__: ", torch.__version__)
    print("torch.version.cuda: ", torch.version.cuda)
    print("torch.cuda.is_available: ", torch.cuda.is_available())
    print("torch.cuda.device_count: ", torch.cuda.device_count())

    a1 = YOLO('yolo11n.pt')

    a1.train(
        data='D:/Users/fuhan/D-Github/rengongzhineng/ultralytics/StudyYolo/data_stjdb.yaml',
        epochs=300,
        imgsz=1024,     # 若仍 OOM，则试 448 或 416
        batch=-1,      # AutoBatch
        device=0,
        workers=0,
        amp=True,
        val=True       # 先留着；要更省资源可改 False
    )

    print('训练完成')



############################## 这是一次训练所有的 #################################

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
