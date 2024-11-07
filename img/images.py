import os
import numpy as np
import cv2

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def file_extension(file: str) -> str:
    return file.split('.')[-1]

def find(folder, file_types: str = None) -> list[str]:
    cwd = os.getcwd()
    folder = os.path.join(cwd, folder)
    picture_files = []
    print("Current working directory: {0}, files: ".format(cwd))
    for file in os.listdir(folder):
        if file_types is None or file_extension(file) in file_types:
            print(file)
            picture_files.append(os.path.join(folder, file))
    return picture_files


def load(picture_files: list[str]) -> list[np.ndarray]:
    """ Load images from files
    get:
        picture_files - list of files
    return:
        images - list of images
    """
    images = []
    for file in picture_files:
        bgr = cv2.imread(file)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(rgb)
    return images