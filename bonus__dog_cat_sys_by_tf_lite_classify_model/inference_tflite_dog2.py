import time
import numpy as np
from tflite_runtime.interpreter import Interpreter  ##這個框架可以乘載訓練好的模型
import cv2
from lite_lib import load_labels, set_input_tensor, classify_image  ##32bit device rpi use tensorflow lite  ##set_input_tensor是不是沒用到?

def gaussian_noise(img, mean=0, sigma=0.1):
    
    # int -> float (標準化)
    img = img / 255
    
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)

    # noise + 原圖
    gaussian_out = img + noise

    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    
    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)

    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)
    
    return gaussian_out

#根據order做不同圖像處理
def img_processing(image, order):
    if order ==0: 
        print("horizon")
        po_image = cv2.flip(image, 1) ##水平翻轉
        cv2.imwrite("horizon.jpg", po_image)
    if order==1:
        print("vertical")
        po_image = cv2.flip(image, 0) ##垂直翻轉
        cv2.imwrite("vertical.jpg", po_image)
    if order ==2: 
        print("add_noise")
        po_image = gaussian_noise(image) ##加入雜訊
        cv2.imwrite("add_noise.jpg", po_image)
    if order==3:
        print("rectangular_mask")
        h, w, c = image.shape[0], image.shape[1], image.shape[2]
        w_mask = w/2
        #print(round(w_mask))
        image[:, 0:round(w_mask), :] = np.zeros((h, round(w_mask), c),np.uint8)
        po_image = image
        cv2.imwrite("rectangular_mask.jpg", po_image)
    if order==4:  ## use gray level img  //onlY 2D, SO INVOKE ERROR    ValueError: could not broadcast input array from shape (255,255) into shape (255,255,3)
        print("binary")  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, pppo_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) ##二值化
        cv2.imwrite("binary.jpg", pppo_image)

        #use to correspond to spec. of API set_input_tensor, which needs 3D input, not 2D
        #SO I need to make gray level img into 3D, BUT LOOK SAME as 2D.
        po_image = np.zeros((pppo_image.shape[0], pppo_image.shape[1], 3))
        po_image[:,:,0] = pppo_image[:,:]
        po_image[:,:,1] = pppo_image[:,:]
        po_image[:,:,2] = pppo_image[:,:]
        cv2.imwrite("binary22.jpg", po_image)
        #print(po_image.shape)
        #REF.  https://discuss.tensorflow.org/t/how-to-predict-in-grayscale-image-with-tensorflow-lite/3837/3
    if order==5:
        print("do nothing")
        po_image = image
        cv2.imwrite("do_nothing.jpg", po_image)
    cv2.waitKey(3000)  #停留3秒
    return po_image

def main(label_path, model_path, img_name, order):
    labels = load_labels(label_path)  #
    interpreter = Interpreter(model_path)  #使用這個助教訓練好的模型產生Interpreter obj.
    print("Model Loaded Successfully.")

    interpreter.allocate_tensors()  ##呼叫訓練好的模型使用的kernels或layers(裡面的weights已經訓練好了)
    _, height, width, channel = interpreter.get_input_details()[0]['shape']  ##模型預設的圖片規格
    print("Required input Shape ({}, {}, {})".format(height, width, channel))

    # load image for inference & preprocessing.
    # the data are normalized to 0~1 whrn training,
    # remenber to do it when inference
    image_show = cv2.imread(img_name)  #
    #if order !=4:
    #    image = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    if order ==5:
        image = image / 255.0  # <- change the pixel range from 0~255 to 0~1

    ## other img processing
    image = img_processing(image, order)  #自己修改order


    # run inference on input image & measure the time spent
    results = classify_image(interpreter, image)  # inference first time  ##為什麼第一次不計時?
    start_time = time.time()
    results = classify_image(interpreter, image)  # inference second time  ##classify_image()會將輸入圖片與模型的kernel做mapping，以得到輸出的分類結果
    stop_time = time.time()
    label_id, prob = results[0]

    # print predict result~
    print(50 * "=")
    print("Object in {} is a/an...".format(img_name))
    print("{}! Confidence={}".format(labels[label_id], prob))
    print(50 * "=")
    print("Time spend: {:.5f} sec.".format(stop_time - start_time))

    # show image
    #cv2.imshow('img', image_show)
    #cv2.waitKey(3000)  #讓結果圖停留3秒
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    label_path = 'model/catdog_label.txt'  ###
    model_path = 'model/catdog_mobilenetv2.tflite'  ###  ##助教訓練好的模型
    img_name = 'catdog_subset/cat2.jpg'  ###  # <- you can change to any other test sample in "cifar10_subset" folder
    for i in range(0,6):
        main(label_path, model_path, img_name, order=i)
    print('Bye')