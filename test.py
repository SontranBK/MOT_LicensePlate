import math
import os
import sys
import threading
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QUrl, QPoint, QRect
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPen, QColor, QPixmap
from numpy import ma
import csv
from user_interface import Ui_MainWindow
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QLabel, QMainWindow, QTableWidgetItem
from moviepy.editor import *

import cv2
import queue
from collections import deque
import time



def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def update(self):
        global vertical
        cap = cv2.VideoCapture(self.fileDir)
        while cap.isOpened():
            start = time.time()
            if self.stop is True or not cap.isOpened():
                self.stop = True
                break

            ret, img = cap.read()
            img_height, img_width, img_colors = img.shape
            scale = self.screen / img_height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            current_color = colorDetector(img, rect)
            WRONG_LANE_X_SIGN = current_color
            current_frame = cap.get(1)
            fps = cap.get(5)
            self.current_time = int(current_frame / fps)
            time_stamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
            box_detects, scores, classes = Detect(detector, img)
            ind = [i for i, item in enumerate(classes) if item == 0]
            classes = np.delete(classes, ind)
            box_detects = np.delete(box_detects, ind, axis=0)
            scores = np.delete(scores, ind)
            data_track = tracker.update(box_detects, scores, classes)
            labels = detector.names

            for i in range(len(data_track)):
                box = data_track[i][:4]
                track_id = int(data_track[i][4])
                cls_id = int(data_track[i][5])
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                self.current[track_id] = midPoint(x0, y0, x1, y1)
                color = (_COLORS[track_id % 30] * 255).astype(np.uint8).tolist()
                text = labels[cls_id] + "_" + str(track_id)
                txt_color = (0, 0, 0) if np.mean(_COLORS[track_id % 30]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                txt_bk_color = (_COLORS[track_id % 30] * 255 * 0.7).astype(np.uint8).tolist()
                if self.ui.ShowLabel.isChecked():
                    cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
                    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
                if self.ui.ShowBox.isChecked():
                    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                if track_id in self.previous:
                    # print(track_id)
                    cv2.line(img, self.previous[track_id], self.current[track_id], color, 1)
                    line_group0 = [lane_left, lane_center, lane_right]
                    for element in line_group0:
                        if len(element) > 0:
                            start_line = element[0][0].x(), element[0][0].y()
                            end_line = element[0][1].x(), element[0][1].y()
                            if intersect(self.previous[track_id], self.current[track_id], start_line, end_line):
                                if line_group0.index(element) == 0:
                                    self.t_counter1.append(track_id)
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_left.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)
                                elif line_group0.index(element) == 1:
                                    self.t_counter2.append(track_id)
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_center.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)
                                elif line_group0.index(element) == 2:
                                    self.t_counter3.append(track_id)
                                    if current_color == BLOW_THE_X_SIGN and not self.ui.red_light_right.isChecked():
                                        self.updateCrossLight(img, x0, y0, track_id, time_stamp)

                    '''
                    t_counter1: line 1
                    t_counter2: line 2
                    t_counter3: line 3
                    t_counter4: line 4
                    t_counter5: line 5
                    t_counter6: line 6
                    '''
                    line_group1 = [direct_left, direct_center, direct_right]
                    for element in line_group1:
                        if len(element) > 0:
                            start_line = element[0][0].x(), element[0][0].y()
                            end_line = element[0][1].x(), element[0][1].y()
                            # print(self.ui.go_forward_left_lane.isChecked())
                            if intersect(self.previous[track_id], self.current[track_id], start_line, end_line):

                                '''
                                xét tại thời điểm cắt qua line 4
                                '''
                                if line_group1.index(element) == 0:
                                    self.t_counter4.append(track_id)
                                    self.ui.Tcounter_3.setText(str(len(list(set(self.t_counter4)))))
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter2 and \
                                                not self.ui.turn_left_center_lane.isChecked():
                                            violation = "sai_lan_giua"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                                        elif track_id in self.t_counter3 and \
                                                not self.ui.turn_left_right_lane.isChecked():
                                            violation = "sai_lan_phai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                                '''
                                xét tại thời điểm cắt qua line 5
                                '''
                                if line_group1.index(element) == 1:
                                    self.t_counter5.append(track_id)
                                    self.ui.Tcounter_4.setText(str(len(list(set(self.t_counter5)))))
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter1 and \
                                                not self.ui.go_forward_left_lane.isChecked():

                                            violation = "sai_lan_trai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)
                                        elif track_id in self.t_counter3 and \
                                                not self.ui.go_forward_right_lane.isChecked():
                                            violation = "sai_lan_phai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                                '''
                                xét tại thời điểm cắt qua line 6
                                '''
                                if line_group1.index(element) == 2:
                                    self.t_counter6.append(track_id)
                                    self.ui.Tcounter_5.setText(str(len(list(set(self.t_counter6)))))
                                    if current_color == WRONG_LANE_X_SIGN:
                                        if track_id in self.t_counter1 and \
                                                not self.ui.turn_right_left_lane.isChecked():
                                            violation = "sai_lan_trai"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)
                                        elif track_id in self.t_counter2 and \
                                                not self.ui.turn_right_center_lane.isChecked():
                                            violation = "sai_lan_giua"
                                            self.updateWrongLane(img, x0, y0, violation, track_id, time_stamp)

                    self.ui.Tcounter_0.setText(str(len(list(set(self.t_counter1)))))
                    self.ui.Tcounter_1.setText(str(len(list(set(self.t_counter2)))))
                    self.ui.Tcounter_2.setText(str(len(list(set(self.t_counter3)))))
                self.previous[track_id] = self.current[track_id]
            end = time.time()
            print('Update {:.5f} seconds'.format(end - start))
            wd01 = QtImage(self.screen, img)
            self.ui.Camera_view.setImage(wd01)