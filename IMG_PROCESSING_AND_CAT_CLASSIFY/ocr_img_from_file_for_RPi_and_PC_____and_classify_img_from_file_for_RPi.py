import time
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

#reset為1則清除檔案內容在輸入其他資料夾檔案
def write_many_line_to_a_file(file_PATH, list_of_text_to_append, reset):

    if reset==1:
        clear_file(file_PATH)

    with open(file_PATH, "w+",encoding="utf-8") as f1:
        #print('writing file now............')
        f1.seek(0)

        for line in list_of_text_to_append:
            f1.write(line)
    # 關閉檔案
    f1.close()
    return None

#以隨機的方式產生雜訊
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
def img_processing(image, order, img_name, output_img_parent_path):
    #print(50 * "=")
    print(">>>>>>start img processing......")
    #start_time = time.time()
    if order ==0: 
        print(">>>horizon")
        #print(image.shape)
        po_image = cv2.flip(image, 1) ##水平翻轉
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_horizon.jpg", po_image)
    if order==1:
        print(">>>vertical")
        po_image = cv2.flip(image, 0) ##垂直翻轉
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_vertical.jpg", po_image)
    if order ==2: 
        print(">>>add_noise")
        po_image = gaussian_noise(image) ##加入雜訊
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_add_noise.jpg", po_image)
    if order==3:
        print(">>>rectangular_mask")
        h, w, c = image.shape[0], image.shape[1], image.shape[2]
        w_mask = w/2
        #print(round(w_mask))
        image[:, 0:round(w_mask), :] = np.zeros((h, round(w_mask), c),np.uint8)
        po_image = image
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_rectangular_mask.jpg", po_image)
    if order==4:  ## use gray level img  //onlY 2D, SO INVOKE ERROR    ValueError: could not broadcast input array from shape (255,255) into shape (255,255,3)
        print(">>>binary")  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, pppo_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) ##二值化
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_binary.jpg", pppo_image)

        #use to correspond to spec. of API set_input_tensor, which needs 3D input, not 2D
        #SO I need to make gray level img into 3D, BUT LOOK SAME as 2D.
        po_image = np.zeros((pppo_image.shape[0], pppo_image.shape[1], 3))
        po_image[:,:,0] = pppo_image[:,:]
        po_image[:,:,1] = pppo_image[:,:]
        po_image[:,:,2] = pppo_image[:,:]
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_binary22.jpg", po_image)
        #print(po_image.shape)
        #REF.  https://discuss.tensorflow.org/t/how-to-predict-in-grayscale-image-with-tensorflow-lite/3837/3
    if order==5:
        print(">>>do nothing")
        po_image = image
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_do_nothing.jpg", po_image)
    if order==6: #blur
        print(">>>bilateralFilter blur to reduce Noise by  11 17 17")
        po_image = cv2.bilateralFilter(image, 11, 17, 17) #blur to reduce Noise
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_bilateralFilter_by_11_17_17.jpg", po_image)
    if order==7: #blur
        print(">>>bilateralFilter blur to reduce Noise by  5 50 100")
        po_image = cv2.bilateralFilter(image, 5, 50, 100)  # smoothing filter
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_bilateralFilter_by_5_50_100.jpg", po_image)
    if order==8: #blur
        print(">>>bilateralFilter blur to reduce Noise by  15 75 75")
        po_image = cv2.bilateralFilter(image, 15, 75, 75)
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_bilateralFilter_by_15_75_75.jpg", po_image)
    if order==9: #blur
        print(">>>bilateralFilter blur to reduce Noise by  15 75 75 and convert to gray level")
        image = cv2.bilateralFilter(image, 15, 75, 75)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, pppo_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) ##二值化

        #use to correspond to spec. of API set_input_tensor, which needs 3D input, not 2D
        #SO I need to make gray level img into 3D, BUT LOOK SAME as 2D.
        po_image = np.zeros((pppo_image.shape[0], pppo_image.shape[1], 3))
        po_image[:,:,0] = pppo_image[:,:]
        po_image[:,:,1] = pppo_image[:,:]
        po_image[:,:,2] = pppo_image[:,:]

        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_bilateralFilter_by_15_75_75_and_turn_gray_level.jpg", po_image)
    if order==10:
        print(">>>GaussianBlur")
        po_image = cv2.GaussianBlur(image, (5, 5), 0) #blur to reduce Noise
        print(po_image.shape)
        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_GaussianBlur.jpg", po_image)
#    if order==7:
#        print("change printed words") ### change printed words as the img processing you have done
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        ### write your img processing here---------------------
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###
#        ###--------------------------------------------------- 
#
#        ### change img name as the img processing you have done
#        cv2.imwrite(output_img_parent_path + name_img_by_current_time() + img_name[:-4] + "_change_img_name.jpg", po_image)
        
    cv2.waitKey(3000)  #停留3秒
    #stop_time = time.time()
    #print("Time spend: {:.5f} sec.".format(stop_time - start_time))
    #print(50 * "=")
    return po_image

#以time.ctime()去得出存有當前時間的list
#並以當前時間命名mp4檔，並回傳。
# image name = a1 + a2 + "_" + a3 + a4 + ".mp4"
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

    img_name_head = a1 + a2 + "_" + a3 + a4 + "_"
    #print("img_name: {}".format(img_name))
    return img_name_head

def main(label_path, model_path, img_name, order, FUNC, classify, img_parent_path, output_img_parent_path):
    image_path = img_parent_path + "/" + img_name
    image = cv2.imread(image_path)  #

    #load model and img preprocessing 
    #------------------------------------------------------------------------------------------------------------------------
    if FUNC==0:  #preprocessing of OCR
        (width, height) = (64, 64)
        #(width, height) = (224, 224)
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
    if FUNC==1:  #preprocessing of classify
        if classify ==0: #RPi
            labels = load_labels(label_path)  #
            interpreter = Interpreter(model_path)  #使用這個助教訓練好的模型產生Interpreter obj.
            #print("Model Loaded Successfully.")

            interpreter.allocate_tensors()  ##呼叫訓練好的模型使用的kernels或layers(裡面的weights已經訓練好了)

            #(height, width, channel) = (224, 224, 3)
            _, height, width, channel = interpreter.get_input_details()[0]['shape']  ##模型預設的圖片規格
            #print("Required input Shape ({}, {}, {})".format(height, width, channel))

            print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            if order ==5:
                image = image / 255.0  # <- change the pixel range from 0~255 to 0~1
        if classify ==1: #PC
            labels = load_labels_CLASS_1(label_path)  #
            model = load_model(model_path, compile=False)  #使用這個助教訓練好的模型產生Interpreter obj.
            #print("Model Loaded Successfully.")

            #(height, width, channel) = (224, 224, 3)
            _, height, width, channel = model.layers[0].output_shape[0]
            #print("Required input Shape ({}, {}, {})".format(height, width, channel))

            #print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            if order ==5:
                image = image / 255.0  # <- change the pixel range from 0~255 to 0~1
            
    ##img processing
    image = img_processing(image, order, img_name, output_img_parent_path)

    image = np.reshape(image, (1, height, width, channel))
    #------------------------------------------------------------------------------------------------------------------------

   

    #detection
    #------------------------------------------------------------------------------------------------------------------------
    if FUNC==0: #run OCR(only on RPi, because module tf-lite only run in RPi)
        custom_config = '-l eng --oem 3 --psm 12'
        #text = pytesseract.image_to_string(gray, config=custom_config, lang="chi_tra")    #中文辨識
        text = pytesseract.image_to_string(gray, lang="chi_tra", config=custom_config, nice=0)  #中英混雜辨識

        file_path2 = str(pathlib.Path().absolute()) + "/OCR_output_txt/" + img_name[:-4] + ".txt"
        write_many_line_to_a_file(file_path2, str(text), 0)#write result to .txt

    if FUNC==1: #run classify(can operate on RPi and PC)
        if classify ==0: #RPi
            # run inference on input image & measure the time spent
            results = classify_image(interpreter, image)  # inference first time  ##為什麼第一次不計時?
            #   start_time = time.time()
            results = classify_image(interpreter, image)  # inference second time  ##classify_image()會將輸入圖片與模型的kernel做mapping，以得到輸出的分類結果
            #   stop_time = time.time()
            label_id, prob = results[0]

            # print predict result~
            #print(50 * "=")
            print("Object in {} is a/an...".format(img_name))
            print("{}! \nConfidence={}".format(labels[label_id], prob))

            prob_CAT = results[3]  #INDEX KNOW FROM LABEL.TXT
            print("\nand\ncat Confidence={}".format(prob_CAT))
            #print(50 * "=")
            #   print("Time spend: {:.5f} sec.".format(stop_time - start_time))
        if classify ==1: #PC
            results = model.predict(image)[0]  # inference first time
            #   start_time = time.time()
            results = model.predict(image)[0]  # inference second time
            #   stop_time = time.time()
            label_id = np.argmax(results)
            prob = results[label_id]

            #print(50 * "=")
            print("Object in {} is a/an...".format(img_name))
            print("{}! \nMOST-LIKE Confidence={}".format(labels[label_id], prob))

            prob_CAT = results[3]  #INDEX KNOW FROM LABEL.TXT
            print("\nand\ncat Confidence={}".format(prob_CAT))
            #print(50 * "=")
            #   print("Time spend: {:.5f} sec.".format(stop_time - start_time))


def read_all_imgs(img_parent_path):
    files_and_directory_names = os.listdir(img_parent_path)
    photo_name = [ele   for ele in files_and_directory_names   if ".jpg" in ele or ".png" in ele] 
    return photo_name

def load_labels_CLASS_1(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

#if __name__ == "__main__":
#    photo_name = read_all_imgs()
#    for K in range(0,1):  ##運行OCR & CLASSIFY
#        for ele in photo_name:  ##輸入所有圖片
#            img_name = str(ele)  ###  # <- you can change to any other test sample in "cifar10_subset" folder
#            for PROCESSING_ORDER in range(0,7):  #決定做那些圖像處理
#                if K==0:  ##OCR
#                    import pytesseract
#                    main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=0)
#                if K==1:  ##CLASSIFY
#                    for G in range(0,1):  #ON RPi or PC
#                        if G==0: #RPi
#                            from tflite_runtime.interpreter import Interpreter  ##這個框架可以乘載訓練好的模型
#                            from lite_lib import load_labels, set_input_tensor, classify_image  ##32bit device rpi use tensorflow lite
#                            label_path = 'model/catdog_label.txt'  ###
#                            model_path = 'model/catdog_mobilenetv2.tflite'
#                            main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=G)
#                        if G==1: #PC
#                            import tensorflow as tf
#                            from tensorflow import keras  ##PC、docker、colab用keras
#                            from tensorflow.keras.models import load_model
#                            label_path = 'model/catdog_label.txt'
#                            model_path = 'model/catdog_mobilenetv2.h5'
#                            main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=G)
#        if K==0:
#            print('OCR DONE\n\n')
#        if K==1:
#            print('OCR DONE\n\n')

if __name__ == "__main__":
    img_parent_path = './catdog_subset'
    output_img_parent_path = './output_img/'  #output_img_parent_path
    photo_name = read_all_imgs(img_parent_path)
    for K in range(0,2):  ##運行OCR & CLASSIFY
        for ele in photo_name:  ##輸入所有圖片
            img_name = str(ele)  ###  # <- you can change to any other test sample in "cifar10_subset" folder
            for PROCESSING_ORDER in range(0,11):  #決定做那些圖像處理
                #if K==0:  ##OCR
                #    import pytesseract
                #    main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=0, img_parent_path, output_img_parent_path)
                if K==1:  ##CLASSIFY
                    for G in range(0,2):  #ON RPi or PC
                        #if G==0: #RPi
                        #    from tflite_runtime.interpreter import Interpreter  ##這個框架可以乘載訓練好的模型
                        #    from lite_lib import load_labels, set_input_tensor, classify_image  ##32bit device rpi use tensorflow lite
                        #    label_path = 'model/catdog_label.txt'  ###
                        #    model_path = 'model/catdog_mobilenetv2.tflite'
                        #    main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=G, img_parent_path, output_img_parent_path)
                        
                        if G==1: #PC
                            if G==1 & PROCESSING_ORDER==0:
                                print("do PC classify")
                            #print(tf.__version__)
                            import tensorflow as tf
                            from tensorflow import keras  ##PC、docker、colab用keras
                            from tensorflow.keras.models import load_model
                            label_path = 'model/cifar10_label.txt'
                            model_path = 'model/cifar10_mobilenetv2.h5'
                            print(50 * "=")
                            main(label_path, model_path, img_name, order=PROCESSING_ORDER, FUNC=K, classify=G, img_parent_path=img_parent_path, output_img_parent_path=output_img_parent_path)
                            print(50 * "=")
        #if K==0:
        #    print('OCR DONE\n\n')
        if K==1:
            print('classify DONE\n\n')