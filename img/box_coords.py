import numpy as np
DEBUG = False

def file_to_xy_box(file_path):
    """ 
    Read file with box coordinates and return list of tuples ((x1, y1), (x2, y2))

    File format:\n
    x1 y1 x2 y2 \n
    x1 y1 x2 y2 \n
    ...

    Returns:
        list of tuples ((x1, y1), (x2, y2))
    """
    NUMBER_OF_VALUES = 4
    def parse_line_to_ints(line):
        values = line.split()
        return [int(value) for value in values]

    def is_text_file(file_path):
        return file_path.endswith('.txt')
    
    if not is_text_file(file_path):
        raise ValueError("File is not text file")
    
    lines = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        numbers =  parse_line_to_ints(line)
        if len(numbers) != NUMBER_OF_VALUES:
            raise ValueError("Line has not 4 numbers")
        yield ((numbers[0], numbers[1]), (numbers[2], numbers[3]))

def load_files_to_dict(files) -> dict:
    dict_files_boxes = {}
    for file in files:
        dict_files_boxes[file] = list(file_to_xy_box(file))
    return dict_files_boxes

def sigma_to_side(sigma):
    return 4*sigma

def convert_blob_to_box(blob):
    x, y, sigma = blob
    side = sigma_to_side(sigma)
    x1 = int(x-side/2)
    x2 = int(x+side/2)
    y1 = int(y-side/2)
    y2 = int(y+side/2)

    x1 = x1 if x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0
    return ((x1, y1), (x2, y2))

def calculate_are_of_intersection(box1:tuple, box2:tuple) -> float:
    """
    ((x1, y1), (x2, y2))
    """
    LEFT = 0
    RIGHT = 1
    X = 0
    Y = 1
    
    def area(box):
        dx = box[RIGHT][X] - box[LEFT][X]
        dy = box[RIGHT][Y] - box[LEFT][Y]
        if dx < 0 or dy < 0:
            return 0
        return dx * dy
    
    # Intersection
    def intersection(box1, box2):
        left = (max(box1[LEFT][X], box2[LEFT][X]), max(box1[LEFT][Y], box2[LEFT][Y]))
        right = (min(box1[RIGHT][X], box2[RIGHT][X]), min(box1[RIGHT][Y], box2[RIGHT][Y]))
        return (left, right)
    
    intersection_area = area(intersection(box1, box2))
    union_area = area(box1) + area(box2) - intersection_area

    if DEBUG:
        print()
        print("Box" + str(box1), "Box" + str(box2))
        print(" + box1 area: ", area(box1))
        print(" + box2 area: ", area(box2))
        print("Intersection Box: ", intersection(box1, box2))
        print(" - Intersection area: ", intersection_area)
        print("Union area: ", union_area)

    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def is_recognized(AoI:float, AoI_treshold:float=0.5) -> bool:
    if AoI > AoI_treshold:
        return True
    return False

def calculate_center(box:tuple) -> tuple:
    """
    ((x1, y1), (x2, y2))
    """
    X = 0
    Y = 1
    x = (box[0][X] + box[1][X]) / 2
    y = (box[0][Y] + box[1][Y]) / 2
    return (x, y)

def calculate_center_distance(center1:tuple, center2:tuple) -> float:
    """
    (x1, y1), (x2, y2)
    """
    X = 0
    Y = 1
    dx = center1[X] - center2[X]
    dy = center1[Y] - center2[Y]
    return np.sqrt(dx**2 + dy**2)

def get_centrals(boxes:list) -> list:
    """
    list of tuples ((x1, y1), (x2, y2))
    """
    centrals = []
    for box in boxes:
        centrals.append(calculate_center(box))
    return centrals

def get_closest(centers:list, center:tuple) -> tuple:
    """
    Args:
        centers: list of tuples (x, y)
        center: tuple (x, y)
    """
    # calculate all distances, with numpy
    centers = np.array(centers)
    center = np.array(center)
    distances = np.linalg.norm(centers - center, axis=1)
    # argmin
    closest_index = np.argmin(distances)
    return centers[closest_index], closest_index

def precision_recall(recognized:list, reference:list, IoU:float=0.5) -> tuple:
    """
    Args:
        recognized: list of tuples ((x1, y1), (x2, y2))
        reference: list of tuples ((x1, y1), (x2, y2))
    
    Returns:
        tuple (precision, recall)
    """
    reference_count = len(reference)
    recognized_count = len(recognized)

    reference_centrals = get_centrals(reference)
    recognized_centrals = get_centrals(recognized)
    
    TP = 0
    FP = 0
    for recognized_center, recognized_box in zip(recognized_centrals, recognized):
        closest, index = get_closest(reference_centrals, recognized_center)
        AoI = calculate_are_of_intersection(recognized_box, reference[index])
        if is_recognized(AoI, IoU):
            TP += 1
        else:
            FP += 1
    
    precision = 0 if reference_count == 0 else TP / reference_count
    recall = 0 if recognized_count == 0 else TP / recognized_count
    return precision, recall

if __name__ == "__main__":
    DEBUG = False

    # test calculate_are_of_intersection

    box1 = ((0, 0), (10, 10))
    box2 = ((5, 5), (15, 15))
    box3 = ((15, 15), (25, 25))
    box4 = ((0, 0), (10, 6))

    print(calculate_are_of_intersection(box1, box2))
    print(calculate_are_of_intersection(box1, box3))
    print(calculate_are_of_intersection(box1, box4))