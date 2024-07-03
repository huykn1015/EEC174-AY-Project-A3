import cv2
import numpy as np
from tracker import Tracker
from math import sqrt

MAX_PATH_PTS = 15
MAX_PATH_LEN = 50
def b2c(box):
    return (int(box[0] + (box[2] / 2)), int(box[1] + (box[3] / 2)))


def edist(p1, p2):
    xa, ya = p1
    xb, yb = p2 
    return sqrt((xa-xb)**2 + (ya-yb)**2)

def draw_line(frame, path, color_box, thickness):
    path = path[::-1]
    start = path[0]
    prev = path[0]
    for i, pt in enumerate(path):
        if  i < MAX_PATH_PTS and edist(start, pt) < MAX_PATH_LEN:
            cv2.line(frame, prev, pt, color_box, thickness)
            prev = pt

# Load YOLO model
network = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg","yolo_files/yolov3.weights")
yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# Load video
video_path = 'input.webm'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = Tracker()

# Determine the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up the video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
out_vid = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))

# Process each frame
for _ in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # - Convert frame to blob
    input_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # - Use YOLO net to detect objects
    network.setInput(input_blob)
    output = network.forward(yolo_layers)
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = frame.shape[:2]

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
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    results = list(filter(lambda x:classes[x] == 0, results.flatten()))
    persons = {b2c(bounding_boxes[i]):bounding_boxes[i] for i in results}
    #print(persons)
    tracker_res = [(res[0], res[1]) for res in tracker.update(persons.keys())]
    if len(tracker_res) > 0 and len(results) > 0:
        for tracker_id, tracked_centroid in tracker_res:
            bounding_box = persons[tracked_centroid]
            colour_box = [int(j) for j in colours[tracker_id]]
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), colour_box, 5)
            cv2.circle(frame, (tracked_centroid[0], tracked_centroid[1]), 10, colour_box, -1)
            text = '%d'%(tracker_id)
            cv2.putText(frame, text, (tracked_centroid[0] - 5, tracked_centroid[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, .8, colour_box, 2)
            draw_line(frame, tracker.get_path(tracked_centroid), colour_box, 5)

    # - Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
    # - Update the tracker with the centroids of detected objects
    # - Draw bounding boxes, trails, and labels on the frame

    # Write the processed frame to the output video
    out_vid.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video writer and video capture objects
out_vid.release()
cap.release()
cv2.destroyAllWindows()
