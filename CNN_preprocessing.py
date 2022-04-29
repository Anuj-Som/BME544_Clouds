# This file will preprocess the gan_image dump into F1-F2 concatenated array data and F3 labels. 
import json
import cv2
import os
from itertools import chain
import numpy as np


def get_images(dir):
    # assign directory
    directory = dir
    
    # iterate over files in
    # that directory
    path_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        
        if os.path.isfile(f):
            path_list.append(f)
    return path_list

def resize_image(img):
    # Resize image to 256x256
    width = 256*2
    height = 256
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def main():
    imgs = get_images("./gan_data/clouds")
    # imgs holds array of every path to image
    print(type(imgs))
    print("Number of images: {}".format(len(imgs)))

    imgs.sort()

    # Enforce imgs is multiple of 3
    rem = len(imgs) % 3
    print(rem)
    imgs = imgs[:len(imgs)-rem]

    # Separate each image stack
    F1s = imgs[0::3]
    F2s = imgs[1::3]
    F3s = imgs[2::3]

    print(len(F1s))
    print(len(F2s))
    print(len(F3s))

    for i in range(len(F1s)):
        # Get frames
        F1 = cv2.imread(F1s[i])
        F2 = cv2.imread(F2s[i])
        F12 = np.concatenate((F1, F2), axis=1)
        cv2.imwrite("./cnn_data/data/data{}.jpg".format(i), F12)     # save frame as JPEG file      
        F3 = cv2.imread(F3s[i])
        cv2.imwrite("./cnn_data/labels/label{}.jpg".format(i), F3)     # save frame as JPEG file      
    print("All triplets converted to data/labels")

if __name__ == "__main__":
    main()