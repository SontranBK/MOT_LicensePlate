import cv2

cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN)
import sys

import numpy as np

sys.path.insert(0, 'Detection')
sys.path.insert(0, 'Tracking')
from Detection.yolov5.detect import Yolov5
from Tracking.bytetrack import BYTETracker

detector = Yolov5(list_objects=["bicycle", "car", "truck", "motorcycle"])
tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                      match_thresh=0.8, min_box_area=10, frame_rate=30)


def Detect(detector, frame):
    box_detects, classes, confs = detector.detect(frame.copy())
    return np.array(box_detects).astype(int), np.array(confs), np.array(classes)


cam = cv2.VideoCapture("/home/haobk/street.mp4")

while 1:
    _, frame = cam.read()

    box_detects, scores, classes = Detect(detector, frame)

    data_track = tracker.update(
        box_detects, scores, classes)
    for data in data_track:
        box = data
        track_id = int(data[4])
        cls_id = int(data[5])
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
'''
(x0,y0) :bottom right coordinates of bbox
x1=x0+width
y1=y0+height
output detect:x0,y0,x1,y1,class,confident score
output tracker:x0,y0,x1,y1,id,class,buffer memory contain 5 (tlwh)
'''