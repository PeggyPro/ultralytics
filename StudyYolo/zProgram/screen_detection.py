# 检测当前屏幕内容
import cv2
import numpy as np
from PIL import ImageGrab
import time
from ultralytics import YOLO

def capture_screen():
    """捕获当前屏幕截图"""
    # 获取整个屏幕截图
    screenshot = ImageGrab.grab()
    # 转换为OpenCV格式 (BGR)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot_cv

def main():
    """主函数：实时检测屏幕内容"""
    print("开始屏幕检测...")
    print("按 'q' 键退出程序")

    # 加载YOLO模型
    model = YOLO('../dnf.pt')  # 使用训练好的模型

    # 设置检测参数
    conf_threshold = 0.5
    imgsz = 640

    try:
        while True:
            # 捕获屏幕
            screen_img = capture_screen()

            # 使用YOLO进行检测
            results = model.predict(
                source=screen_img,
                conf=conf_threshold,
                imgsz=imgsz,
                verbose=False  # 不打印详细信息
            )

            # 处理检测结果
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # 在屏幕上绘制检测框
                annotated_img = r.plot()

                # 显示结果
                cv2.imshow('Screen Detection', annotated_img)

                # 打印检测信息
                print(f"\n检测到 {len(boxes)} 个目标:")
                for i in range(len(boxes)):
                    # 获取坐标
                    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = r.names[cls_id]

                    print(f"  - {cls_name} {conf:.2f} [{x1}, {y1}, {x2}, {y2}]")

            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # 控制检测频率，避免CPU占用过高
            time.sleep(0.1)  # 100ms间隔

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        cv2.destroyAllWindows()
        print("屏幕检测结束")

def single_screen_detection():
    """单次屏幕检测"""
    print("执行单次屏幕检测...")

    # 加载模型
    model = YOLO('../dnf.pt')

    # 捕获屏幕
    screen_img = capture_screen()

    # 保存原始截图
    cv2.imwrite('screen_capture.png', screen_img)
    print("屏幕截图已保存为: screen_capture.png")

    # 进行检测
    results = model.predict(
        source=screen_img,
        conf=0.5,
        imgsz=640,
        save=True,  # 保存检测结果
        project='screen_detection_results'  # 保存到指定文件夹
    )

    # 打印检测结果
    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("未检测到任何目标")
            continue

        print(f"\n检测到 {len(boxes)} 个目标:")
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            cls_name = r.names[cls_id]

            print(f"  - {cls_name} {conf:.2f} [{x1}, {y1}, {x2}, {y2}]")

    print("检测完成！结果已保存到 screen_detection_results 文件夹")

if __name__ == "__main__":
    print("屏幕检测程序")
    print("1. 实时屏幕检测")
    print("2. 单次屏幕检测")

    choice = input("请选择模式 (1/2): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        single_screen_detection()
    else:
        print("无效选择，默认执行单次检测")
        single_screen_detection()
