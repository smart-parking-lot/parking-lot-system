#程式在接觸感測器被碰觸時(傳high給pin 11)使用 PiCamera 拍下影像並進行風格轉換的處理
import cv2
import time
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera

#拍攝並存取照片(名稱會是以當前時間命名，但前面有capture_這段字串)，然後進入風格轉換的函式處理
def camera_setup_and_shot():
    try:
        img_name = "capture_" + name_img_by_current_time()
        print("capture picture!!!!")
        camera.capture(img_name)    # 拍照並儲存
        img = cv2.imread(img_name)  # 讀取
        img = cv2.flip(img,0)#上下翻轉
        #img = cv2.flip(img,0)#上下翻轉
        #img = cv2.flip(img,1)#左右翻轉
        cv2.imwrite(img_name, img) #重新儲存轉正後的照片
        print("save capture img!!!!")
        cv2.imshow('capture', img)   # 將 img 輸出至視窗
        #change_style(img)
        # cv2.waitKey(100)            #dalay 100ms  # 等待按下鍵盤或 100 ms

    except KeyboardInterrupt:
        print('interrupt')

    #finally:
        #camera.close()
        #cv2.destroyAllWindows()

#當gpio input pin 11接收到high(接觸感測器被觸碰)，則會拍攝並進行風格轉換
#def change_style(img):
#    print("start changing style.....................\n")
#    parent_path = "/home/pi/Desktop/lab2/hw2-1/"
#    net = cv2.dnn.readNetFromTorch(parent_path + 'starry_night.t7') # 讀取風格檔，這裡讀入「星空」的風格  #記得修改成你的風格檔的path
#    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#
#    (h, w) = img.shape[:2]
#
#    # 把影像修改成神經網路可以使用的格式
#    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
#
#    net.setInput(blob)  # 把影像丟入模型做風格轉換
#    out = net.forward() # 開始轉換
#    print("finish changing style!!!!")
#    out = out.reshape(3, out.shape[2], out.shape[3])
#    out[0] += 103.939
#    out[1] += 116.779
#    out[2] += 123.68
#    # out /= 255
#    out = out.transpose(1, 2, 0)
#    cv2.imshow('output', out)   # 將 img 輸出至視窗
#    img_name = name_img_by_current_time()
#    cv2.imwrite(img_name, out)
#    print("save output!!!!")


#以time.ctime()去得出存有當前時間的list
#並以當前時間命名jpg檔，並回傳。
# image name = a1 + a2 + "_" + a3 + a4
def name_img_by_current_time():
    #---------use current time to name img----------------------------------------------------------
    A = time.ctime()
    A = A.split()
    #print("value of time.ctime(): {}".format(A))
    #['Mon', 'Oct', '25', '07:16:50', '2021']

    #image name = a1 + a2 + "_" + a3 + a4
    a1='0'
    mon = {'Oct':'10', 'Nov':'11' , 'Dec':'12', 'Jan':'01' , 'Feb':'02', 'Mar':'03' , 'Apr':'04', 'May':'05' , 'Jan':'06', 'Jul':'07' , 'Aug':'08', 'Sep':'09'}

    for (key, value) in mon.items():
        if A[1]==key:
            a1=value
            #print("a1 = {}".format(a1))

    a2 = A[2]

    TMP = A[3]
    a3 = TMP[0:2]
    a4 = TMP[3:5]

    img_name = a1 + a2 + "_" + a3 + a4 + ".jpg"
    #print("img_name: {}".format(img_name))
    return img_name
    #-------------------------------------------------------------------------------------------------



#driver program
#當接觸感測器被觸碰，會傳1給RPi pin 11，然後螢幕顯示"touch"，並進入相機環節
#---------------------------------------------------------------------------------
#gpio input及camera resolution 的 config
SWITCH_PIN = 27
#GPIO.setmode(GPIO.BOARD) #定義模式，只要用到GPIO都需要定義模式
GPIO.setmode(GPIO.BCM) 
GPIO.setup(SWITCH_PIN, GPIO.IN) #告訴RPi，上述腳位們是input or output
time.sleep(0.1) #warm up

camera = PiCamera()
camera.resolution = (320, 240)
time.sleep(0.1) #warm up

try:
    while True:
        if GPIO.input(SWITCH_PIN) == GPIO.HIGH:  #當接觸感測器被觸碰
            print("Touch-->camera")
            camera_setup_and_shot() #shot and change style
            print("ALL done\n\n")
        else:  #當接觸感測器沒被觸碰         # if GPIO.input(SWITCH_PIN) == GPIO.LOW:  
            print("No Touch")

        time.sleep(1) #休息1秒後再進行下一輪判斷
        
except KeyboardInterrupt:
    print("kb")
finally:
    GPIO.cleanup() #程式結束，記得釋放腳位
#---------------------------------------------------------------------------------