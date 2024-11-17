# By Pytel

import numpy as np
import cv2


def img2hue_histogram(img: np.ndarray) -> np.ndarray:
    """ Convert RGB image to Hue histogram
    get:
        img - RGB image
    return:
        hist - Hue histogram
    """
    # RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]

    # Histogram
    #max_value = 256
    max_value = 180
    hist, b = np.histogram(hue, bins=max_value, range=(0, max_value))

    # normalize histogram
    hist = hist / np.max(hist)

    return hist


def get_center_of_picture(picture: np.ndarray) -> tuple():
    """ Get center of picture
    get:
        picture - 2D numpy array
    return:
        x_center, y_center - center of picture
    """
    pixel_sum = np.sum(picture)

    weight_x = 0
    weight_y = 0
    for x in range(picture.shape[1]):
        for y in range(picture.shape[0]):
            weight_x += picture[y, x] * x
            weight_y += picture[y, x] * y

    x_center = weight_x / pixel_sum
    y_center = weight_y / pixel_sum
    return x_center, y_center


class CamShift():
    """ CamShift class """

    def __init__(self, pattern_file) -> None:
        """ Init
        get:
            pattern_file - path to pattern file
        """
        patern_bgr = cv2.imread(pattern_file)
        patern_rgb = cv2.cvtColor(patern_bgr, cv2.COLOR_BGR2RGB)
        y, x, z = patern_rgb.shape
        self.x_size = x
        self.y_size = y

        self.pattern_hue_hist = img2hue_histogram(patern_rgb)

        self.last_positon = None

    def _get_first_positon(self, img) -> tuple():
        """ Get first position of pattern in image
        get:
            img - BGR image
        return:
            x_center, y_center - center of pattern
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]

        # img projection
        hue_projection = self.pattern_hue_hist[hue]

        # get center of hue projection
        x_center, y_center = get_center_of_picture(hue_projection)
        return x_center, y_center

    def _get_next_positon(self, next_img) -> tuple():
        """ Get next position of pattern in image
        get:
            next_img - BGR image
        return:
            x_center, y_center - center of pattern
        """
        x1, y1 = self.last_positon[0]
        x2, y2 = self.last_positon[1]
        crop_img = next_img[y1:y2, x1:x2]

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]

        hue_projection = self.pattern_hue_hist[hue]

        x_center, y_center = get_center_of_picture(hue_projection)
        x_center += x1
        y_center += y1
        return x_center, y_center

    def next_positon(self, next_img) -> tuple():
        if self.last_positon == None:
            x_center, y_center = self._get_first_positon(next_img)
        else:
            x_center, y_center = self._get_next_positon(next_img)

        x1 = int(x_center - self.x_size/2)
        y1 = int(y_center - self.y_size/2)
        x2 = int(x_center + self.x_size/2)
        y2 = int(y_center + self.y_size/2)

        self.last_positon = ((x1, y1), (x2, y2))
        return ((x1, y1), (x2, y2))
