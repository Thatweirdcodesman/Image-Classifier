import cv2
import numpy as np
from modules.resizeImage import resize_image


def crop_image(im):
    img = im
    # print(type(img))
    # cv2.imshow("original", img)
    blurred = cv2.blur(img, (6, 6))
    # cv2.imshow("blurred",blurred)
    canny = cv2.Canny(blurred, 50, 200)
    # cv2.imshow("canny", canny)

    # find the non-zero min-max coords of canny
    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    # print('min', x1, y1)
    # print('max', x2, y2)

    # crop the region
    cropped = img[y1:y2, x1:x2]
    # cv2.imwrite("/home/hariharan/Downloads/cropped.png", cropped)
    # cv2.imshow("cropped", cropped)

    crop_canny = cv2.Canny(cropped, 50, 200)
    # cv2.imshow("canny", crop_canny)

    resized = resize_image(cropped)
    # cv2.imshow("resized", resized)
    # print('resized shape:', resized.shape)
    tagged = cv2.rectangle(img.copy(), (x1,y1), (x2,y2), (0,255,0), 3, cv2.LINE_AA)
    return resized
    # cv2.imshow("tagged", tagged)
    # cv2.waitKey()

