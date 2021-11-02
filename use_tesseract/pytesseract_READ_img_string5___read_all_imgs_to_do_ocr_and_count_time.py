import pytesseract
from PIL import Image
import pathlib
import time
import cv2
from matplotlib import pyplot as plt
import os

def name_txt_by_origin_name_add_output_head(img_name):
    img_name = img_name[:-4] + ".txt"
    return img_name

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

def read_ocr_and_write_txt(img_name):
    #recognition character in the img
    #img =Image.open ("1.png")
    #img_name = "image23.png"  #修改成要測試的圖片
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    #plt.show()
    #alpha, beta = 1.7, 0   #沒有幫助
    #gray = cv2.convertScaleAbs(alpha, beta)  #沒有幫助
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

    parent_path = str(pathlib.Path().absolute())
    file_name_2 = "/" + name_txt_by_origin_name_add_output_head(img_name)
    file_path2 = parent_path + file_name_2
    write_many_line_to_a_file(file_path2, str(text), 0)#write result to .txt


def read_all_img_names():
    #透過這個path找到當前路徑的所有檔案及資料夾名稱
    files_and_directory_names = os.listdir()

    #讓新list只有"圖片檔"的名稱
    photo_names = [ele   for ele in files_and_directory_names   if ".jpg" in ele or ".png" in ele] 
    #print(photo_name)
    return photo_names

for ele in read_all_img_names():
    print('\n>>>>>>>Start Processing:{}.........\n'.format(ele))
    start_time = time.time()  #紀錄起始時間
    #print(str(ele))
    read_ocr_and_write_txt(ele)
    end_time = time.time()
    print('\n>>>>>>>>{} Time cost:{}\n'.format(ele, end_time - start_time))
#////////////////////////////////////////////////////////////////////////////