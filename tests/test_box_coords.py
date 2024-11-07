import sys
import os
import pytest

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from img.box_coords import *

@pytest.mark.parametrize("box1, box2, expected", [
    (((0, 0), (10, 10)), ((5, 5), (15, 15)), 25),
    (((0, 0), (10, 10)), ((15, 15), (25, 25)), 0),
    (((0, 0), (10, 10)), ((0, 0), (10, 6)), 60),
])
def test_of_intesection_area(box1, box2, expected):
    assert area(intersection(box1, box2)) == expected

@pytest.mark.parametrize("blob, expected_box", [
    ((5, 5, 5), ((0, 0), (15, 15))),
    ((5, 5, 10), ((0, 0), (25, 25))),
    ((0, 20, 5), ((0, 10), (10, 30))),
])
def test_convert_blob_to_box(blob, expected_box):
    assert convert_blob_to_box(blob) == expected_box
