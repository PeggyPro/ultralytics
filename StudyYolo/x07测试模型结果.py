# 检测模型结果
import time

from ultralytics import YOLO

# 模型训练完毕自动保存到: runs/detect/train/weights
# best.pt 是训练好的最优模型(使用于最终应用)
# last.pt 是训练的最后一轮模型(使用于继续训练)

if __name__ == '__main__':
    start_time = time.time()
    print("start time: {}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # 加载自己训练好的模型
    a1 = YOLO(r'D:\Users\fuhan\D-Github\rengongzhineng\ultralytics\学习yolov\runs\detect\train5\weights\best.pt')

    # 目标检测
    a1('16.jpg', show=False, save=True)

    # predict() 返回的是一个列表, 列表中的每个元素是一个对象, 对象中包含了检测到的目标信息
    results = a1.predict('16.jpg')
    class_names = a1.names  # 获取模型的类别名称列表

    for result in results:
        class_indices = result.boxes.cls  # 获取每个框的类别索引
        class_names_for_boxes = [class_names[int(idx)] for idx in class_indices]  # 获取每个框的类别名称

        xywh = result.boxes.xywh  #
        xywhn = result.boxes.xywhn
        print(class_names_for_boxes)
        print(xywh)
        print(xywhn)

    print("end time: {}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("used time : {}", time.time() - start_time)
