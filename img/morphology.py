# By Pytel
# https://core.ac.uk/download/pdf/44387378.pdf

import numpy as np
from enum import Enum, auto

from my_libs.img.functional import *

class MorphologyOperation(Enum):
    GRAY_ERODE = auto()
    GRAY_DILATE = auto()
    GRAY_OPEN = auto()
    GRAY_CLOSE = auto()
    GRAY_TOP_HAT = auto()
    GRAY_ERODE_1D = auto()
    GRAY_DILATE_1D = auto()
    GRAY_OPEN_1D = auto()
    GRAY_CLOSE_1D = auto()

def gray_erode_1D(img, kernel):
    sumed_kernel = np.sum(kernel, axis=0, dtype=int)
    out = np.zeros_like(img)
    for i_x in range(img.shape[0]):
        for i_y in range(img.shape[1]):
            values = []
            for k_x in range(sumed_kernel.shape[0]):
                x = i_x + k_x
                y = i_y
                if valid_coord(x, y, img):
                    delta = int(img[x, y]) - sumed_kernel[k_x]
                    values.append(0 if delta < 0 else delta)
            out[i_x, i_y] = np.min(values)
    return out

def gray_dilate_1D(img, kernel):
    sumed_kernel = np.sum(kernel, axis=0, dtype=int)
    out = np.zeros_like(img)
    for i_x in range(img.shape[0]):
        for i_y in range(img.shape[1]):
            values = []
            for k_x in range(sumed_kernel.shape[0]):
                x = i_x - k_x
                y = i_y
                if valid_coord(x, y, img):
                    delta = int(img[x, y]) + sumed_kernel[k_x]
                    values.append(255 if delta > 255 else delta)
            out[i_x, i_y] = np.max(values)
    return out

def gray_erode(img, kernel):
    """
    Uřezávám vršky
    min{f(x + z) - k(z)}; z e K; x + z e F
    """
    I_X, I_Y = img.shape
    K_X, K_Y = kernel.shape
    out = np.zeros_like(img)
    img_padded = cv2.copyMakeBorder(img, 1, K_X-1, 1, K_Y-1, cv2.BORDER_CONSTANT, value=255)
    for i_x in range(I_X):
        for i_y in range(I_Y):
            out[i_x, i_y] = np.min(img_padded[i_x:i_x+K_X, i_y:i_y+K_Y] - kernel)
    return out

def gray_dilate(img, kernel):
    """
    Zaplácávám díry
    max{f(x - z) + k(z)}; z e K; x - z e F
    """
    I_X, I_Y = img.shape
    K_X, K_Y = kernel.shape
    out = np.zeros_like(img)
    img_padded = cv2.copyMakeBorder(img, K_X-1, 1, K_Y-1, 1, cv2.BORDER_CONSTANT, value=0)
    for i_x in range(I_X):
        for i_y in range(I_Y):
            out[i_x, i_y] = np.max(img_padded[i_x:i_x+K_X, i_y:i_y+K_Y] + kernel)
    return out

"""
def dilate(img_gray, kernel):
    output = np.zeros_like(img_gray)

    for i in range(1,img_gray.shape[0]-1):
        for j in range(1, img_gray.shape[1]-1):
            temp = img_gray[i:i+2, j-1:j+2]
            output[i, j] = np.max(temp*kernel)
    return output[1:-1,1:-1]

def erode(img_gray, kernel):
    output = np.zeros_like(img_gray)

    for i in range(1,img_gray.shape[0]-1):
        for j in range(1, img_gray.shape[1]-1):
            temp = img_gray[i:i+2, j-1:j+2]
            output[i, j] = np.min(temp/kernel)
    return output[1:-1,1:-1]
"""

def morphology(img, operation : MorphologyOperation, kernel):
    if operation == MorphologyOperation.GRAY_ERODE_1D:
        return gray_erode_1D(img, kernel)
    elif operation == MorphologyOperation.GRAY_DILATE_1D:
        return gray_dilate_1D(img, kernel)
    elif operation == MorphologyOperation.GRAY_OPEN_1D:
        eroded = gray_erode_1D(img, kernel)
        dilarated = gray_dilate_1D(eroded, kernel)
        return dilarated
    elif operation == MorphologyOperation.GRAY_CLOSE_1D:
        dilarated = gray_dilate_1D(img, kernel)
        eroded = gray_erode_1D(dilarated, kernel)
        return eroded
    
    if operation == MorphologyOperation.GRAY_ERODE:
        return gray_erode(img, kernel)
    elif operation == MorphologyOperation.GRAY_DILATE:
        return gray_dilate(img, kernel)
    elif operation == MorphologyOperation.GRAY_OPEN:
        eroded = gray_erode(img, kernel)
        dilarated = gray_dilate(eroded, kernel)
        return dilarated
    elif operation == MorphologyOperation.GRAY_CLOSE:
        dilarated = gray_dilate(img, kernel)
        eroded = gray_erode(dilarated, kernel)
        return eroded
    elif operation == MorphologyOperation.GRAY_TOP_HAT:
        """
        f(x) - (f(x) o B)
        img - opened(img, kernel)
        """
        eroded = gray_erode(img, kernel)
        opened = gray_dilate(eroded, kernel)
        
        out = np.zeros(img.shape, dtype=np.uint8)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                delta = int(img[x, y]) - int(opened[x, y])
                if delta < 0:
                    out[x, y] = 0
                elif delta > 255:
                    out[x, y] = 255
                else:
                    out[x, y] = delta
        return out.astype(np.uint8)
    else:
        raise Exception("Unknown operation")
