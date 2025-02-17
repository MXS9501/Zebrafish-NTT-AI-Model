import cv2

# 选择第二个摄像头，索引为1
cap2 = cv2.VideoCapture(0)

# 设置摄像头分辨率
width = 640  # 设置宽度
height = 360  # 设置高度
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 检查第二个摄像头是否成功打开
if not cap2.isOpened():
    print("无法打开第二个摄像头")
    exit()

while True:
    # 读取第二个摄像头的一帧图像
    ret2, frame2 = cap2.read()

    # 检查是否成功读取第二个摄像头的帧
    if not ret2:
        print("无法获取第二个摄像头的帧")
        break

    # 显示第二个摄像头的帧
    cv2.imshow('Camera 2', frame2)

    # 按 'q' 键退出循环
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放第二个摄像头并关闭所有窗口
cap2.release()
cv2.destroyAllWindows()
