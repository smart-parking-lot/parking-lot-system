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