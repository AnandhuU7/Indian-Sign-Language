import cv2
import numpy as np
import os


def nothing(x):
    pass


image_x, image_y = 64, 64

from keras.models import load_model

classifier = load_model('ISLModel.h5')

fileEntry = []
for file in os.listdir("SampleGestures"):
    if file.endswith(".png"):
        fileEntry.append(file)

def predictor():
    from keras.preprocessing import image
    test_image = image.load_img('output.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)


    result = classifier.predict(test_image)

    print(result)


    for i in range(len(fileEntry)):
        image_to_compare = cv2.imread("./SampleGestures/" + fileEntry[i])
        original = cv2.imread("output.png")
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(original, None)
        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        ratio = 0.6
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)
        if (abs(len(good_points) + len(matches)) > 50):
            gesname = fileEntry[i]
            gesname = gesname.replace('.png', '')
            return gesname


    classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','CALL','D','E','F','G','H','HELP','I','J','K','L','M','N','O','P','Q','R','S','T','THANK_YOU','U','V','W','WIN','X','Y','Z']
    max_index = np.argmax(result)
    return classes[max_index]

cam = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
window_width = 500
window_height = 300
cv2.resizeWindow("Trackbars", window_width, window_height)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 160, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 160, 255, nothing)
cv2.namedWindow("ISL Recognition")

img_text = ''
# img_text1 = ''

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    # cv2.putText(frame, img_text1, (80, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("ISL Recognition", frame)
    cv2.imshow("mask", mask)

    img_name = "output.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()

    if cv2.waitKey(1) == 27:
        break




cam.release()
cv2.destroyAllWindows()
