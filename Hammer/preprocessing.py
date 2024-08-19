import cv2
import numpy as np
import os

dir = 'images\\train'

def preprocess_image(rawImg):
    img = cv2.imread(rawImg)
    # Convert the image to grayscale
    p = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    p = cv2.GaussianBlur(p, (13, 13), 0)
    p = cv2.medianBlur(p, 11)
    _, p = cv2.threshold(p, 250, 255, cv2.THRESH_BINARY_INV)
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    k3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    p = cv2.erode(p,k1)
    p = cv2.copyMakeBorder(
        p,
        top=25,
        bottom=25,
        left=25,
        right=25,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN,k3)
    p = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k2)
    p = cv2.morphologyEx(p, cv2.MORPH_OPEN, k3)
    q = cv2.erode(p,k4)
    p = p-q
    return p

def preprocess_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            imgPath = os.path.join(folder_path, filename)
            pImg = preprocess_image(imgPath)
            cv2.imshow(filename + '_Contours', pImg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

preprocess_images_in_folder(dir)