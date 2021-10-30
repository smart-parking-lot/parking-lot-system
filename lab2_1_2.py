from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

camera = PiCamera()
camera.resolution = (540, 540)
time.sleep(0.1)

try:
    while True:
        camera.capture('1.jpg')    # 拍照並儲存至 1.jpg
        img = cv2.imread('1.jpg')  # 讀取 1.jpg
        img = cv2.flip(img,0)#上下翻轉
        img = cv2.flip(img,1)#上下翻轉
        cv2.imshow('capture', img)   # 將 img 輸出至視窗
        cv2.waitKey(1000)            #dalay 1000ms  # 等待按下鍵盤或 1000 ms

except KeyboardInterrupt:
    print('interrupt')

finally:
    camera.close()
    cv2.destroyAllWindows()
