def img_processing_func(img):
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(img, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)
    plot.imshow(binary,cmap="gray")
    plot. show()

    ret, binary = cv2.threshold(binary, 250, 255, cv2.THRESH_BINARY)
    plot.imshow(binary, cmap="gray")
    plot.show()

    image, contour, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    copy_img = img.copy()
    for cnt An contour:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(copy_img, (x,y), (x+w, y+h), (0,255,0), 40)
    plot. imshow(copy_img)
    plot. show()

    cnt = max(contour, key=len)
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = binary[y+100:y+h-100, x+100:x+W-100]
    plot. imshow(crop_img, cmap="gray")
    plot .show()

    crop_x, crop_y = crop_img.shape
    print(crop_x, crop_y)
    new_img = cv2.resize(crop_img, (int(crop_y/10), int(crop_x/10)))
    plot. imshow(new_img, cmap="gray")
    plot .show()

    blur = cv2.medianBlur(new_img,5)
    plot.imshow(blur, cmap="gray")
    plot.show()
