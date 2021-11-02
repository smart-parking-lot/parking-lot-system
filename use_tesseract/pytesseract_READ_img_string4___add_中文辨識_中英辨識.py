import pytesseract
from PIL import Image
import pathlib
import time
import cv2
from matplotlib import pyplot as plt

#//////////////////////////////write char into .txt
#以time.ctime()去得出存有當前時間的list
#並以當前時間命名mp4檔，並回傳。
# image name = a1 + a2 + "_" + a3 + a4 + ".mp4"
def name_txt_by_current_time():
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

    img_name = a1 + a2 + "_" + a3 + a4 + ".txt"
    #print("img_name: {}".format(img_name))
    return img_name

def name_txt_by_origin_name_add_output_head(img_name):
    img_name = img_name[:-4] + ".txt"
    return img_name

#reset為1則清除檔案內容在輸入其他資料夾檔案
def write_many_line_to_a_file(file_PATH, list_of_text_to_append, reset):
    if reset==1:
        clear_file(file_PATH)

    with open(file_PATH, "w+",encoding="utf-8") as f1:
        print('writing file now............')
        f1.seek(0)

        for line in list_of_text_to_append:
            f1.write(line)
    # 關閉檔案
    f1.close()
    return None



#main 
#////////////////////////////////////////////////////////////////////////////
#recognition character in the img
#img =Image.open ("1.png")
img_name = "image16.jpg"
img = cv2.imread(img_name, cv2.IMREAD_COLOR)#修改成要測試的圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
#plt.show()
#alpha, beta = 1.7, 0 
#gray = cv2.convertScaleAbs(alpha, beta)  #這段會讓運行時間增加
gray = cv2.bilateralFilter(gray, 11, 17, 17) #blur to reduce Noise
#custom_config = r'--oem 3 --psm 6'
custom_config = '-l eng --oem 3 --psm 12'
#text = pytesseract.image_to_string(gray, config=custom_config, lang="chi_tra")    #中文辨識
text = pytesseract.image_to_string(gray, lang="chi_tra+eng", config=custom_config, nice=0)  #中英混雜辨識
    #lang="chi_tra+eng", config=custom_config這兩個參數會讓運行時間增加
    #Using oem and psm in Tesseract Raspberry Pi for better results
    #lang="chi_tra+eng"實現中文辨識且實現中英文夾雜。
#text = pytesseract.image_to_string(img, config="")
print (text)

#write result to .txt
parent_path = str(pathlib.Path().absolute())
file_name_2 = "/" + name_txt_by_origin_name_add_output_head(img_name)
file_path2 = parent_path + file_name_2
write_many_line_to_a_file(file_path2, str(text), 0)


#////////////////////////////////////////////////////////////////////////////