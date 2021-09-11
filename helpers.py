import pandas as pd
import numpy as np
from skimage import io
from scipy import ndimage
import cv2
import os
import time
import sys
import cv2

    
def pad_image_for_square( img ):
    '''
    Pad image to retain aspect ratio
    INPUT
        img
    OUTPUT
        img
    '''
    (rows, cols, dim) = img.shape
    resize_dim = max( rows, cols)
    pad_row = resize_dim - rows
    pad_col = resize_dim - cols
    img = np.pad(img, ((0,pad_row), (0, pad_col), (0,0)), 'constant', constant_values=0)
    return img


def rotate_images(file_path, dest_path, degrees_of_rotation, lst_imgs):
    '''
    Rotates image based on a specified amount of degrees: 90, 120, 180, 270
    INPUT
        file_path: file path to the folder containing images.
        degrees_of_rotation: Integer, specifying degrees to rotate the
        image. Set number from 1 to 360.
        lst_imgs: list of image strings.
    OUTPUT
        Images rotated by the degrees of rotation specififed.
    '''

    for l in lst_imgs:
        img = cv2.imread(file_path + str(l))

        #rotation angle in degree
        img = ndimage.rotate(img, degrees_of_rotation)
        cv2.imwrite(dest_path + str(degrees_of_rotation) + "_" + str(l), img)


def mirror_images(file_path, dest_path, mirror_direction, lst_imgs):
    '''
    Mirrors image left or right, based on criteria specified.
    INPUT
        file_path: file path to the folder containing images.
        mirror_direction: criteria for mirroring left or right.
        lst_imgs: list of image strings.
    OUTPUT
        Images mirrored left or right.
    '''

    for l in lst_imgs:
        img = cv2.imread(file_path + str(l))
        img = cv2.flip(img, 1)
        cv2.imwrite(dest_path + "mir_"+ str(l), img)

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)



def crop_and_resize_images(path, new_path, img_size=299):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.
    OUTPUT
        All images cropped, resized, and saved from the old folder to the new folder.
    '''

    create_directory(new_path)
    dirs = [l for l in os.listdir(path)[:] if l != '.DS_Store']
    total = 0
    val = 900
    for item in dirs:
        img = cv2.imread(path+item)
        #all the image r in same shape
        img = cv2.resize(img,(1994,1328),interpolation=cv2.INTER_AREA)

        y,x,channel = img.shape
        startx = x//2-(val//2)
        starty = y//2-(val//2)
        img = img[starty:starty+val,startx:startx+val]
        img = cv2.resize(img,(img_size, img_size),interpolation=cv2.INTER_AREA)
        cv2.imwrite(new_path+item, img)
        img = None
        total += 1
        print("Saving: ", item, total)

#------------------------------------------------------------------
def center_crop(data, shape):
   
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize(image, MIN_BOUND, MAX_BOUND):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image, PIXEL_MEAN):
    image = image - PIXEL_MEAN
    return image

'''
MIN_BOUND = np.min(x_test)
MAX_BOUND = np.max(x_test)
PIXEL_MEAN = np.mean(x_test)
MIN_BOUND, MAX_BOUND, PIXEL_MEAN

x_test = normalize(x_test)
x_test = zero_center(x_test)
x_test.shape
'''
