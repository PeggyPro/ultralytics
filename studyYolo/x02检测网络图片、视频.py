# 检测网络图片、视频
from ultralytics import YOLO

########################## 1.简单使用(0摄像头会内存溢出) ############################

# YOLO优势/特点: 按照每帧(1秒 = 60帧)来检测目标
# a1 = YOLO('yolo11n.pt')

# 检测目标可以传入图片、视频(本地、网络), 0摄像头
# a1("https://i0.hdslb.com/bfs/archive/a74136e5a14f5a5a0a9721dbf8d9f400ade61f05.jpg", show=True, save=True)

# a1(0, show=True, save=False)
# a1('1.png', show=True, save=False)


########################## 2.打印详情(内存不会溢出) ############################

model = YOLO('yolo11n.pt')

for r in model.predict(source=0, show=True, stream=True, conf=0.5, imgsz=640):
    boxes = r.boxes  # Boxes 对象
    if boxes is None or len(boxes) == 0:
        continue

    # 每帧的文件/流路径（摄像头一般是 0）
    print(f"\nFrame from: {r.path} | Dets: {len(boxes)}")

    for i in range(len(boxes)):
        # xyxy 坐标（左上 x1,y1，右下 x2,y2）
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cls_id = int(boxes.cls[i].item())     # 类别ID
        conf = float(boxes.conf[i].item())    # 置信度
        cls_name = r.names[cls_id]            # 类别名

        # 如果是 track 模式（model.track），可以打印跟踪ID
        track_id = None
        if getattr(boxes, "id", None) is not None:
            tid = boxes.id[i]
            track_id = int(tid.item()) if hasattr(tid, "item") else int(tid)

        if track_id is None:
            print(f"  - {cls_name} {conf:.2f} [{x1}, {y1}, {x2}, {y2}]")
        else:
            print(f"  - ID {track_id}: {cls_name} {conf:.2f} [{x1}, {y1}, {x2}, {y2}]")
