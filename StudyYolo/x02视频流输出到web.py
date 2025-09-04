from flask import Flask, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# 加载YOLO模型
model = YOLO('stjdb.pt')

# 打开摄像头流
cap = cv2.VideoCapture(0)  # 0为默认摄像头

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 进行YOLO目标检测
        results = model.predict(source=frame, stream=True, conf=0.5, imgsz=640)

        # 处理每帧的结果
        for r in results:
            boxes = r.boxes  # Boxes 对象
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # xyxy 坐标（左上 x1,y1，右下 x2,y2）
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                cls_id = int(boxes.cls[i].item())     # 类别ID
                conf = float(boxes.conf[i].item())    # 置信度
                cls_name = r.names[cls_id]            # 类别名

                # 画出矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{cls_name} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 编码视频帧为JPEG格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>实时视频流</title>
        </head>
        <body>
            <h1>监控视频流</h1>
            <img src="/video_feed" style="width: 640px; height: 480px;">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
