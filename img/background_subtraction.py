import cv2 as cv
from box_coords import create_box_coordinates_from_bin_img

class BackGroundSubtraction:
    def __init__(self, background = None, trashhold = 50):
        self.background = background
        self.trashhold = trashhold

    def calculate_diff(self, frame):
        if self.background is None:
            self.background = frame
            return None

        diff = cv.absdiff(self.background, frame)
        return diff

    def process(self, frame):
        diff = self.calculate_diff(frame)
        # diff to grayscale
        diff_mask = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        # threshold
        _, diff_mask = cv.threshold(diff_mask, self.trashhold, 255, cv.THRESH_BINARY)

        # invert
        #diff_mask = cv.bitwise_not(diff_mask)

        coordinates = create_box_coordinates_from_bin_img(diff_mask)
        return coordinates