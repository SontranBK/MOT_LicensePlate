import os
import cv2 
# cv2.namedWindow("Draw 1 polygon(s)", cv2.WINDOW_NORMAL) 
from EasyROI import EasyROI

import pickle



roi_helper = EasyROI(verbose=True)

url="/home/haobk/video.mp4"
cam=cv2.VideoCapture(url)
for i in range(20):
    _,frame =cam.read()
    
ratio=frame.shape[1]/frame.shape[0]
frame=cv2.resize(frame,(800,int(800/ratio)))
polygon_roi = roi_helper.draw_polygon(frame, 1) # quantity=3 specifies number of polygons to draw
polygon = polygon_roi["roi"][0]["vertices"]
points=[]
data={"polygon":polygon,"shape":frame.shape[:2]}
for i in range(len(polygon)):
    points.append((polygon[i][0]/frame.shape[1],polygon[i][1]/frame.shape[0]))
  

pickle.dump(data,open("data/polygon.pickle","bw+"))
frame_temp = roi_helper.visualize_roi(frame, polygon_roi)
cv2.imwrite("data/polygon.jpg",frame_temp)
