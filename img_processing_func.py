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