# By Pytel

import cv2
import numpy as np
from functional import convolution

def mean(img, kernel_size : int = 3) -> np.ndarray:
    """ Implements 2D mean filter

    Args:
        img : image to convolute
        kernel_size : size of kernel
    Returns:
        new_img : convoluted image
    
    #### Usecases:
    - noise reduction
    - low pass filter

    #### Algorithm:
    The mean filter is a simple linear filter that replaces each pixel with the mean of its neighbors.
    """
    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    return convolution(img, kernel)

def rotation_mask(image, kernel_size : int = 3):
    """ Implements 2D rotation mask filter

    Args:
        img : image to convolute
        kernel_size : size of kernel
    Returns:
        new_img : convoluted image

    #### Usecases:
    - noise reduction
    - low pass filter

    #### Algorithm:
    The rotation mask filter has 1 kernel and 8 different positions.
    Mask is moving around each pixel and searching for the position with the lowest variance.
    Output image is the mean of the pixel in the lowest variance position.
    """
    output_image = np.zeros(image.shape)
    h0,h1 = (kernel_size-1)//2, (kernel_size-1)//2 
    new_image = np.pad(image,((h0,h0),(h1,h1)),'constant',constant_values=(0,0))
    variancies = np.zeros(new_image.shape)
    means = np.zeros(new_image.shape)

    for y, row in enumerate(new_image):
        if(y == 0 or y == new_image.shape[0]):
            continue
        for x, pixel in enumerate(row):
            if (x == 0 or x == new_image.shape[1]):
                continue
            window = new_image[y-1:y+2, x-1:x+2]
            variancies[y, x] = np.var(window)
            means[y, x] = np.mean(window)
            
    for y, row in enumerate(image):
        if(y == 0 or y == image.shape[0]):
            continue
        for x, pixel in enumerate(row):
            if (x == 0 or x == image.shape[1]):
                continue
            window = variancies[y-1:y+2, x-1:x+2]
            window_mean = means[y-1:y+2, x-1:x+2]
            window[1,1] = float('inf')
            tmp = np.unravel_index(window.argmin(), window.shape)
            tmp_min = window_mean[tmp]
            output_image[y,x] = tmp_min
    return output_image

def median(img, kernel_size : int = 3) -> np.ndarray:
    """ Implements 2D median filter 
    
    #### Usecases:
    - noise reduction
    - low pass filter
    """
    X_img, Y_img = img.shape
    X_ker, Y_ker = kernel_size, kernel_size
    
    img = cv2.copyMakeBorder(img, X_ker-1, X_ker-1, Y_ker-1, Y_ker-1, cv2.BORDER_CONSTANT, value=128)
    
    new_img = np.zeros((X_img, Y_img))
    for x in range(X_ker-1, X_img + X_ker-1):
        for y in range(Y_ker-1, Y_img + Y_ker-1):
            """
            # take cros from image
            y_axis = img[x, y-Y_ker+1:y+1]
            x_axis = img[x-X_ker+1:x+1, y]
            # get median
            new_img[x-X_ker+1, y-Y_ker+1] = np.median(np.concatenate((y_axis, x_axis)))
            """
            new_img[x-X_ker+1, y-Y_ker+1] = np.median(img[x-X_ker+1:x+1, y-Y_ker+1:y+1])
    
    # crop image
    out = new_img[X_ker-1:X_img+X_ker-1, Y_ker-1:Y_img+Y_ker-1]
    return out

def laplace(img) -> np.ndarray:
    """ Implements 2D laplace filter
    
    Args:
        img : image to convolute
    Returns:
        new_img : convoluted image

    #### Usecases:
    - edge detection
    - high pass filter

    #### Algorithm:
    The laplace filter is a simple linear filter that replaces each pixel with the laplace of its neighbors.
    
    Mask:
    | | | |
    | --- | --- | --- |
    | 0 | 1 | 0 |
    | 1 | -4| 1 |
    | 0 | 1 | 0 |

    Mask:
    | | | |
    | --- | --- | --- |
    | 1 | 1 | 1 |
    | 1 | -8| 1 |
    | 1 | 1 | 1 |
    """
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]])

    return convolution(img, kernel)

def sobel(img) -> np.ndarray:
    kernels = []
    kernels.append(np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]))
    kernels.append(np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]]))
    kernels.append(np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]))
    kernels.append(np.array([
        [-2, -1, 0],
        [-1, 0, 1],
        [0, 1, 2]]))
    kernels.append(np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]]))
    kernels.append(np.array([
        [0, -1, -2],
        [1, 0, -1],
        [2, 1, 0]]))
    kernels.append(np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]]))
    edges = []
    for kernel in kernels:
        edges.append(convolution(img, kernel))
    edges_np = np.array(edges)
    return np.max(edges_np, axis=0)

def kirsch(img) -> np.ndarray:
    kernels = []
    kernels.append(np.array([
        [-5, -5, -5],
        [3, 0, 3],
        [3, 3, 3]]))
    kernels.append(np.array([
        [-5, -5, 3],
        [-5, 0, 3],
        [3, 3, 3]]))
    kernels.append(np.array([
        [-5, 3, 3],
        [-5, 0, 3],
        [-5, 3, 3]]))
    kernels.append(np.array([
        [3, 3, 3],
        [-5, 0, 3],
        [-5, -5, 3]]))
    kernels.append(np.array([
        [3, 3, 3],
        [3, 0, 3],
        [-5, -5, -5]]))
    kernels.append(np.array([
        [3, 3, 3],
        [3, 0, -5],
        [3, -5, -5]]))
    kernels.append(np.array([
        [3, 3, -5],
        [3, 0, -5],
        [3, 3, -5]]))
    kernels.append(np.array([
        [3, -5, -5],
        [3, 0, -5],
        [3, 3, 3]]))
    edges = []
    for kernel in kernels:
        edges.append(convolution(img, kernel))
    edges_np = np.array(edges)
    return np.max(edges_np, axis=0)
