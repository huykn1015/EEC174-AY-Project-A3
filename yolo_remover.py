import cv2
import numpy as np
import argparse
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


parser = argparse.ArgumentParser(
                    prog='yolo_remover',
                    description='Removes specified object from image',
                    epilog='Text at the bottom of help')
parser.add_argument('image_path', type=str, help='path to image') 
parser.add_argument('rm_object', type=str, help='name of object to remove') 
parser.add_argument('block_size', type=str, help='block size used to write over bounding boxes of objects')
args = parser.parse_args()










LABELS_FILE_PATH = 'yolo_files/coco.names'

def get_class_names(labels_files):
    my_file = open(labels_files, "r") 
    class_names = my_file.read() 
    class_names = class_names.split("\n") 
    my_file.close() 
    return class_names

def get_class_index(class_names, class_name):
    return class_names.index(class_name)
    

class_names = get_class_names(LABELS_FILE_PATH)
if args.rm_object not in class_names:
    print('Object: "%s" not found in available classes'%args.rm_object)
    exit(1)

if not os.path.isfile(args.image_path):
    print('%s not found'%(args.image_path))
    exit(1)


block_size = int(args.block_size)


def remove_row(image, x_start, y_start, width,height, block_size):
    block_y = block_size
    block_x = block_size
    if block_size > height:
        block_y = height
    if block_size > width:
        block_x = width
    img = Image.fromarray(np.uint8(image))
    block = Image.fromarray(np.uint8(image)).crop((x_start + width, y_start + height - block_y, x_start + width + block_x, y_start + height))
    #block.show()
    cur_x = x_start + width - block_x
    while cur_x > x_start: #paste block on image until space remaining to cover is < block_size
        img.paste(block, (cur_x, y_start))
        cur_x -= block_x
    print(cur_x, x_start)
    if cur_x != x_start: #if block is bigger than remaining image to paste on, crop block to remaining space
        block = block.crop((0, 0, cur_x + block_x - x_start, block_y))
        img.paste(block, (x_start, y_start))
    #img.show()
    image = np.array(img)
    return image

def inpaint_yolo_box(image, yolo_boxes, block_size):
    """
    Inpaints specified bounding boxes in the image using a right-to-left copy method.

    Parameters:
    image (numpy.ndarray): The original image.
    yolo_boxes (list of tuples): List of bounding boxes, each defined as (x, y, width, height).
    block_size: BLOCK_SIZExBLOCK_SIZE box to copy over.

    Returns:
    The inpainted image.
    """
    for x_min, y_min, width, height in yolo_boxes:
        for row in range((height) // block_size):
            image = remove_row(image, x_min, y_min + (row * block_size), width, block_size, block_size)
    return image
#def test_block_get():
    #image = cv2.imread('imgs/beach.jpg')
    #print(image.shape)
    #img = inpaint_yolo_box(image, [(150, 400, 950, 200)], 100)

def main(image_path, object_to_remove, block_size):
    # - Load YOLO model
    global class_names
    network = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg","yolo_files/yolov3.weights")
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    # - Load and process the image
    image = cv2.imread(image_path)
    image_name = image_path.split('.')[0]
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(input_blob)
    output = network.forward(yolo_layers)
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]

    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)
    # - Get bounding boxes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    results = list(filter(lambda x:classes[x] == get_class_index(class_names, object_to_remove), results.flatten()))
    rm_boxes = [bounding_boxes[i] for i in results]
    # - Call inpaint_yolo_box on all boxes
    image = inpaint_yolo_box(image, rm_boxes, block_size)
    # - Save inpainted image


    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(image_name + '_out.png')
    pass
main(args.image_path, args.rm_object, block_size)