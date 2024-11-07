# By Pytel

import cv2
import numpy as np

def img_to_g(img) -> np.ndarray:
    """ Convert image to g.

    Args:
        img: image to convert

    Return:
        g: g image
    
    #### Algorithm:
    g = (G*255)/(R+G+B)
    """
    #emps = float(1e-10)
    R = img[:,:,0].astype(float)
    G = img[:,:,1].astype(float)
    B = img[:,:,2].astype(float)
    g = (G*255)/(R+G+B)# + emps)
    return g.astype(np.uint8)

def img_to_r(img) -> np.ndarray:
    """ Convert image to r.

    Args:
        img: image to convert

    Return:
        r: r image
    
    #### Algorithm:
    r = (R*255)/(R+G+B)
    """
    emps = float(1e-10)
    R = img[:,:,0].astype(float)
    G = img[:,:,1].astype(float)
    B = img[:,:,2].astype(float)
    r = (R*255)/(R+G+B + emps)
    return r.astype(np.uint8)

def segmentate(
        img, 
        threshold : np.uint8 = 100, 
        scale : np.uint8 = 255,
        invert : bool = False
    ) -> np.ndarray:
    """ Segmentate image by threshold.

    Args:
        img: image to segmentate
        threshold: threshold value
        scale: scale of weigths (0-1)*scale

    Return:
        img: image with 0 and 255

    #### Algorithm:
    ![gif](https://cse19-iiith.vlabs.ac.in/exp/image-segmentation/images/image001.gif)
    """
    # img : 0-1
    if invert:
        img_segmented = img < threshold
    else:
        img_segmented = img > threshold
    # img : 0-255 <= (0-1)*scale
    img_segmented = img_segmented * scale
    return img_segmented.astype(np.uint8)


def flood_fill(img, img_filled, x : int, y : int, object_number) -> bool:
    """ Flood fill algorithm.

    Args:
        img: image to fill
        img_filled: image with filled points 
        x: x coordinate of start point
        y: y coordinate of start point
        object_number: number of object

    Return:
        True if point is filled, False if not

    #### Algorithm:
    ![gif](https://upload.wikimedia.org/wikipedia/commons/b/b6/Wfm_floodfill_animation_queue.gif)
    """

    # TODO: remove recursion !!!

    X, Y = img.shape
    # 1) check if point is in image
    if x < 0 or x >= X or y < 0 or y >= Y:
        return False
    # 2) check if point is not filled
    if img_filled[x][y] != 0:
        return False
    # 3) check if point is not background
    if img[x][y] == 255:
        return False
    # 4) fill point and call flood_fill for neighbours
    img_filled[x][y] = object_number
    for vector in [(0,1), (0,-1), (1,0), (-1,0)]:
        dx, dy = vector
        flood_fill(img, img_filled, x+dx, y+dy, object_number)
    return True

def color_objects_with_flood_fill(img : np.ndarray) -> tuple():
    """ Collor objects in image by separate numbers.

    Args:
        img: image to collor objects

    Return:
        img: image with collored objects
        object_number: number of objects

    #### Algorithm
    Maximal number of objects is 255.
    Img background value is 255.
    Collor background number is 0.
    """
    X, Y = img.shape
    objects = np.zeros([X, Y], dtype=np.uint8)
    object_number = 1
    for x in range(X):
        for y in range(Y):
            if img[x][y] == 255:
                continue
            elif flood_fill(img, objects, x, y, object_number):
                object_number += 1

    return objects, object_number

def valid_coord(x : int, y : int, img : np.ndarray) -> bool:
    """ Check if coordinates are valid.

    Args:
        x: x coordinate
        y: y coordinate
        img: image to check

    Return:
        True if coordinates are valid, False if not
    """
    X, Y = img.shape
    if x < 0 or x >= X or y < 0 or y >= Y:
        return False
    return True


def color_objects(img : np.ndarray) -> tuple():
    """ Collor objects in image by separate numbers.

    Args:
        img: image to collor objects

    Return:
        img: image with collored objects
        list_of_objects: numbers of objects

    #### Algorithm
    Maximal number of objects is 255.
    Img background value is 255.
    Collor background number is 0.
    """
    MAX_OBJECTS = 255

    def object_is_background(objects, array_of_keys, x, y):
        return objects.get(array_of_keys[x][y], 0) == 0
    
    def get_color(objects, array_of_keys, x, y):
        return objects.get(array_of_keys[x][y], 0)
    
    def int2uint8(x):
        return np.uint8(x % 256)

    def solve_collor_from_neighbours(x : int, y : int, img : np.ndarray, array_of_keys : np.ndarray, objects : dict, object_number : int) -> int:
        """ Collor neighbours of point.

        Args:
            x: x coordinate of point
            y: y coordinate of point
            img: image to collor
            array_of_keys: array of keys
            objects: dictionary of objects
            object_number: number of object

        Return:
            object_number: number of object
        """
        
        # 1) check if point is not background
        if img[x][y] == 255:
            return object_number

        posible_colors = []
        for vector in [(-1,-1), (-1,0), (-1,1), (0,-1)]:
            dx, dy = vector
            if not valid_coord(x+dx, y+dy, img):
                # out of image
                continue
            elif object_is_background(objects, array_of_keys, x+dx, y+dy):
                # neighbour is background
                continue
            elif get_color(objects, array_of_keys, x+dx, y+dy) not in posible_colors:
                # neighbour is colored
                posible_colors.append(get_color(objects, array_of_keys, x+dx, y+dy))
        
        #print("X:", x, "Y:", y, "Posible colors:", posible_colors)
        if len(posible_colors) == 0:
            # new object
            array_of_keys[x][y] = object_number
            objects[object_number] = object_number
            object_number += 1
            #print("New color:", object_number)
        else:
            # color from neighbours
            selected_color = min(posible_colors)
            array_of_keys[x][y] = selected_color
            for color in posible_colors:
                objects[color] = selected_color
        return object_number
    
    objects = {}
    objects[0] = 0
    X, Y = img.shape
    array_of_keys = np.zeros([X, Y])
    object_number = 1
    for x in range(X):
        for y in range(Y):
            object_number = solve_collor_from_neighbours(x, y, img, array_of_keys, objects, object_number)

    # re-number objects
    for number, object in enumerate(objects):
        if number == 0:
            continue
        objects[object] = number

    # array of keys to colored image
    colored = np.zeros([X, Y], dtype=np.uint8)
    object_numbers = {}
    for x in range(X):
        for y in range(Y):
            color = get_color(objects, array_of_keys, x, y)
            colored[x][y] = int2uint8(color)
            object_numbers[color] = color
    
    # remove background from list of objects
    list_of_objects = list(object_numbers.values())
    list_of_objects.sort()
    list_of_objects.remove(0)

    return colored, list_of_objects

def calculate_centers_of_objects(img, object_numbers=[1]) -> dict:
    """ Calculate centers of objects in image.

    Args:
        img: image, where objects are marked by numbers
        object_numbers: list of numbers of objects

    Return: 
        dict of centers
    """
    X, Y = img.shape
    moments = [(1, 0), (0, 1), (0, 0)]

    centers = {}
    for object_number in object_numbers:
        centers[object_number] = [0, 0, 0]

    for x in range(X):
        for y in range(Y):
            number = img[x][y]
            if number not in object_numbers:
                continue
            for index, moment in enumerate(moments):
                centers[number][index] += x**moment[0] * y**moment[1]

    for key in centers:
        centers[key][0] /= centers[key][2]
        centers[key][1] /= centers[key][2]

    return centers


def granulometry(img, kernel_size=100) -> list:
    """
    Granulometry is a morphological operation that measures the size distribution of objects in a binary image.
    It is defined as the difference between the original image and the opening of the image by a series of structuring elements.
    The structuring elements are disks of increasing size.

    Args:
        img : binary image
        kernel_size : size of the kernel
    
    Returns:
        granulometry : list of granulometry values
    """
    granulometry = []
    last = np.copy(img)
    # Iterate over the kernel sizes
    for i in range(1, kernel_size + 1):
        # Create a kernel
        kernel = np.ones((i, i), np.uint8)
        # Perform the opening operation
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Subtract the last opening
        granulometry.append(np.sum(opening - last))
        last = opening        
    return granulometry

def histogram(img) -> np.ndarray:
    """ Calculate histogram of image
    
    Args:
        img : image to calculate histogram
        
    Returns:
        hist : histogram of image
    """
    Y = img.shape[0]
    X = img.shape[1]
    hist = np.zeros([256,1])
    for y in range(Y):
        for x in range(X):
            hist[img[y][x]] +=1
    return hist

def convolution(img, kernel : np.ndarray) -> np.ndarray:
    """ Implements 2D convolution 

    Args:
        img : image to convolute
        kernel : kernel to convolute

    Returns:
        new_img : convoluted image

    #### Algorithm:
    ![gif](https://miro.medium.com/v2/resize:fit:1400/1*Fw-ehcNBR9byHtho-Rxbtw.gif)
    """
    X_img, Y_img = img.shape
    X_ker, Y_ker = kernel.shape
    
    img = cv2.copyMakeBorder(img, X_ker-1, X_ker-1, Y_ker-1, Y_ker-1, cv2.BORDER_CONSTANT, value=128)
    
    new_img = np.zeros((X_img, Y_img))
    for x in range(X_ker-1, X_img + X_ker-1):
        for y in range(Y_ker-1, Y_img + Y_ker-1):
            new_img[x-X_ker+1, y-Y_ker+1] = np.sum(np.multiply(img[x-X_ker+1:x+1, y-Y_ker+1:y+1], kernel))

    return new_img

def intenzity_corecton(img, etalon) -> np.ndarray:
    """ Corect image intenzity
    
    Args:
        img : image to corect
        etalon : etalon image
        
    Returns:
        img : corected image

    #### Algorithm:
    img = img / etalon
    """
    return (img/etalon).astype(np.uint8)

def normalize(img) -> np.ndarray:
    """ Normalize image to 0-255.

    Args:
        img: image to normalize
    
    Return:
        img: normalized image

    #### Algorithm:
    img = (img - min(img)) / (max(img) - min(img)) * 255
    """
    img_min = np.min(img)
    img_max = np.max(img)
    img = img - img_min
    img = img / (img_max - img_min)
    img = img * 255
    return img.astype(np.uint8)

# TODO: use P0
def ekvalise(img, q0 = 0, p0 = 0, qk = 255) -> np.ndarray:
    """ Ekvalise image
    
    Args:
        img : image to ekvalise
        q0 : start value
        p0 : start value
        qk : end value
    
    Returns:
        ekvalised : ekvalised image
    """
    def sumed(array) -> np.ndarray:
        ret = np.zeros(array.shape[0])
        sumed = 0
        for i in range(array.shape[0]):
            sumed += array[i]
            ret[i] = sumed
        return ret 

    Y = img.shape[0]
    X = img.shape[1]
    hist = histogram(img)
    #hist = np.histogram(img, bins=256)[0]
    sumed_values = sumed(hist)
    coef = (qk-q0)/(X*Y)
    ekvalised = np.zeros([Y,X])
    for y in range(Y):
        for x in range(X):
            ekvalised[y][x] = coef * sumed_values[img[y][x]]
    return ekvalised.astype('uint8')
