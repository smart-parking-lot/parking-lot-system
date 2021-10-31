#將資料夾的圖片都改成imagex.jpg，x是個數字。
import os
import cv2

#透過這個path找到一系列的檔案及資料夾名稱
parent_dir = "C:\Users\user\Downloads\img"
files_and_directory_names = os.listdir(parent_dir)

#根據有無substring ".jpg"和".png"判斷是否放入新生成的list
#讓新list只有"圖片檔"的名稱
photo_name = [ele   for ele in files_and_directory_names   if ".jpg" or ".png" in ele] 

#讀取指定名稱的圖片，並改成imagexx.jpg的名稱
i=0
index=list(range(8,100))  #改這裡
for ele in photo_name:
    img = cv2.imread(ele)
    out_image_name = "image" + str(index[i]) + ele[-4:] 
    cv2.imwrite(out_image_name, img)
    i+=1