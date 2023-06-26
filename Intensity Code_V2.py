import os
import cv2
import numpy as np

from numpy.linalg import norm


def brightness(img):
    top = 250
    right = 1150
    height = 900
    width = 1600
    img = img[top:(top + height), right: (right + width)]
    # # Display cropped image
    # cv2.imshow("cropped", img)
    # convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower bound and upper bound for Yellow color
    lower_bound = np.array([20, 50, 50])
    upper_bound = np.array([50, 255, 255])

    # find the colors within the boundaries
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # define kernel size
    kernel = np.ones((7, 7), np.uint8)

    # Remove unnecessary noise from mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("mask", mask)

    # Segment only the detected region
    segmented_img = cv2.bitwise_and(hsv, hsv, mask=mask)
    # cv2.imshow("seg", segmented_img[:, :, 2])

    # Find contours from the mask

    # contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

    # cv2.imshow("seg",segmented_img)

    # Showing the output

    # cv2.imshow("Output", output)

    #  calculate histogram of third channel V, which is the brightness.

    # cv2.imshow("Output2", mask.copy()[..., 2])

    # The function waitKey waits for a key event infinitely (when \f$\texttt{delay}\leq 0\f$ ) or for delay milliseconds, when it is positive
    # cv2.waitKey(0)

    # The function destroyAllWindows destr  oys all of the opened HighGUI windows.
    cv2.destroyAllWindows()

    yellow_brightness = np.sum(segmented_img[:, :, 2])
    return yellow_brightness


if __name__ == '__main__':

    root_dir = ''
    neg_folder_name = 'Neg/'
    pos_folder_name = 'Pos/'

    for image in os.listdir(root_dir + pos_folder_name):
        if ".jpg" in image or ".png" in image:
            img = cv2.imread(root_dir + pos_folder_name + "/" + image)
            # if image == "83-1.jpg":
            #     print("img")
            bri = brightness(img)
            # print('POS',  bri[0],bri[1],bri[2] ,image, sep='\t')
            print('POS', bri, image, sep='\t')
        # print()
    for image in os.listdir(root_dir + neg_folder_name):
        if ".jpg" in image or ".png" in image:
            img = cv2.imread(root_dir + neg_folder_name + "/" + image)
            bri = brightness(img)
            # print('NEG',  bri[0],bri[1],bri[2] ,image, sep='\t')
            print('NEG', bri, image, sep='\t')
            print(brightness(img))