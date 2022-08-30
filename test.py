import cv2
cv2.namedWindow("frame",cv2.WINDOW_FULLSCREEN)
import sys

import numpy as np
sys.path.insert(0, 'Detection')
sys.path.insert(0, 'Tracking')
from Detection.yolov5.detect import Yolov5
from Tracking.bytetrack import BYTETracker
detector = Yolov5(list_objects=["bicycle","car","truck","motorcycle"])
tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                            match_thresh=0.8, min_box_area=10, frame_rate=30)

def Detect( detector, frame):
    box_detects, classes, confs = detector.detect(frame.copy())
    return np.array(box_detects).astype(int), np.array(confs), np.array(classes)
cam=cv2.VideoCapture("/home/haobk/street.mp4")


while 1:
    _,frame=cam.read()

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

        
        color = (255, 255, 255)
        
        trace=data[6]
       
        for pos in trace:
            frame = cv2.circle(frame, (int(pos[0]+pos[2]/2),int(pos[1]+pos[3]/2)), 1, color, 3)
   
        text = str(track_id)
        txt_color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (255, 255, 255)
        cv2.rectangle(
            frame,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(
            frame, text, (x0, y0 + txt_size[1]), font, 0.5, txt_color, thickness=2)
    cv2.imshow("frame",frame)
    cv2.waitKey(1)
    
     