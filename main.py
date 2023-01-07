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

from Detection.yolox.detect import YoloX
from Tracking.bytetrack import BYTETracker
from Detection.yolox.utils.visualize import _COLORS

ROOT_DIR = os.path.abspath(os.curdir)
ICON_PATH = os.path.join(ROOT_DIR, 'static/icon.png')
detector = YoloX()
tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                      match_thresh=0.8, min_box_area=10, frame_rate=30)
# -*- coding: utf-8 -*-

q = queue.Queue()
pts = deque(maxlen=64)
qp = QtGui.QPainter()
boundaries = [[((0, 110, 95), (20, 255, 255))],  # Red
              [((20, 90, 60), (50, 200, 200))],  # Yellow
              [((80, 90, 110), (90, 150, 255))]]  # Green

global rect
global lane_left, lane_center, lane_right
global direct_left, direct_center, direct_right
global vertical



BLOW_THE_X_SIGN = 'green'
WRONG_LANE_X_SIGN = 'green'


def getFrame(video_dir, myFrameNumber):
    cap = cv2.VideoCapture(video_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
    cap.grab()
    return cap.retrieve(0)


def QtImage(screen, img):
    img_height, img_width, img_colors = img.shape
    scale = screen / img_height
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    height, width, bpc = img.shape
    bytesPerLine = 3 * width
    return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)


def Detect(det, frame):
    box_detects = det.detect(frame.copy())[0]
    classes = det.detect(frame.copy())[1]
    confs = det.detect(frame.copy())[2]
    return np.array(box_detects).astype(int), np.array(confs), np.array(classes)


def Average(lst):
    return sum(lst) / len(lst)


def colorDetector(frame, _rect):
    num_rect = len(_rect)
    # print("xxx",type(_rect))
    color_sum = [0, 0, 0]
    color = []
    for j in range(num_rect):
        a = rect[0][0].y()
        b = rect[0][0].x()
        c = rect[0][1].y()
        d = rect[0][1].x()
        traffic_light = frame[a:c, b:d]
        traffic_light = cv2.GaussianBlur(traffic_light, (5, 5), 0)
        hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
        for i in range(len(boundaries)):
            for (lower, upper) in boundaries[i]:
                mask = cv2.inRange(hsv, lower, upper)
                color_sum[i] = ma.sum(mask)
        color.append(color_sum.index(np.max(color_sum)))
    color_lb = None
    if len(color) > 0:
        average = Average(color)
        if average == 2:
            color_lb = 'green'
        elif average == 0:
            color_lb = 'red'
        else:
            color_lb = 'yellow'
    return color_lb


class CameraView(QWidget):
    def __init__(self, parent=None):
        super(CameraView, self).__init__(parent)
        self.image = None
        global rect
        global lane_left, lane_center, lane_right
        global direct_left, direct_center, direct_right

    def setImage(self, image):
        self.image = image
        img_size = image.size()
        self.setMinimumSize(img_size)
        self.update()

    def paintEvent(self, event):
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        br = QtGui.QBrush(QColor(0, 252, 156, 40))
        qp.setBrush(br)

        for item in rect:
            pen = QPen(QColor(255, 255, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawRect(QRect(item[0], item[1]))

        for item in lane_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 1"))
        for item in lane_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 1"))
        for item in lane_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 1"))

        for item in direct_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 2"))
        for item in direct_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 2"))
        for item in direct_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 2"))
        qp.end()


class ExportView(QWidget):
    def __init__(self, parent=None):
        super(ExportView, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        img_size = image.size()
        self.setMinimumSize(img_size)
        self.update()

    def paintEvent(self, event):
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class DrawObject(QWidget):
    def __init__(self, parent=None):
        super(DrawObject, self).__init__(parent)
        self.image = None
        self.flag = None
        self.direct = None
        self.img_holder = None
        self.screen = None
        global rect
        global lane_left, lane_center, lane_right
        global direct_left, direct_center, direct_right
        rect = []
        lane_left, lane_center, lane_right = [], [], []
        direct_left, direct_center, direct_right = [], [], []
        self.begin = QPoint()
        self.end = QPoint()
        self.show()

    def setImage(self, image):
        self.image = image
        img_size = image.size()
        self.setMinimumSize(img_size)
        self.update()

    def setMode(self, mode_flag, direct_flag):
        self.flag = mode_flag
        self.direct = direct_flag
        self.update()

    def goBack(self):
        if self.flag == 'rect' and len(rect) > 0:
            rect.pop()
        elif self.flag == 'lane' and self.direct == 'left' and len(lane_left) > 0:
            lane_left.pop()
        elif self.flag == 'lane' and self.direct == 'center' and len(lane_center) > 0:
            lane_center.pop()
        elif self.flag == 'lane' and self.direct == 'right' and len(lane_right) > 0:
            lane_right.pop()
        elif self.flag == 'direct' and self.direct == 'left' and len(direct_left) > 0:
            direct_left.pop()
        elif self.flag == 'direct' and self.direct == 'center' and len(direct_center) > 0:
            direct_center.pop()
        elif self.flag == 'direct' and self.direct == 'right' and len(direct_right) > 0:
            direct_right.pop()
        self.update()

    def paintEvent(self, event):
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        br = QtGui.QBrush(QColor(0, 252, 156, 40))
        qp.setBrush(br)

        for item in rect:
            pen = QtGui.QPen(QColor(255, 255, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawRect(QRect(item[0], item[1]))

        for item in lane_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 1"))
        for item in lane_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line center 1"))
        for item in lane_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line right 1"))

        for item in direct_left:
            pen = QtGui.QPen(QColor(255, 0, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("line left 2"))
        for item in direct_center:
            pen = QtGui.QPen(QColor(255, 255, 0), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            qp.drawText(item[0], str("center line 2"))
        for item in direct_right:
            pen = QtGui.QPen(QColor(0, 0, 255), 2, QtCore.Qt.SolidLine)
            qp.setPen(pen)
            qp.drawLine(item[0], item[1])
            # print(item[0], item[1])
            qp.drawText(item[0], str("right line 2"))
        qp.end()

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        if self.direct == 'left':
            if self.flag == 'lane' and len(lane_left) == 1:
                lane_left.pop()
            elif self.flag == 'direct' and len(direct_left) == 1:
                direct_left.pop()
        elif self.direct == 'center':
            if self.flag == 'lane' and len(lane_center) == 1:
                lane_center.pop()
            elif self.flag == 'direct' and len(direct_center) == 1:
                direct_center.pop()
        elif self.direct == 'right':
            if self.flag == 'lane' and len(lane_right) == 1:
                lane_right.pop()
            elif self.flag == 'direct' and len(direct_right) == 1:
                direct_right.pop()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end = event.pos()
        self.update()
        if self.direct == 'left':
            if self.flag == 'lane':
                lane_left.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_left.append([self.begin, self.end])
        elif self.direct == 'center':
            if self.flag == 'lane':
                lane_center.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_center.append([self.begin, self.end])
        elif self.direct == 'right':
            if self.flag == 'lane':
                lane_right.append([self.begin, self.end])
            elif self.flag == 'direct':
                direct_right.append([self.begin, self.end])
        elif self.flag == 'rect':
            rect.append([self.begin, self.end])


def midPoint(a, b, c, d):
    return int(a + (c - a) / 2), int(b + (d - b) / 2)


def shortDir(file_dir):
    stringNum = len(file_dir)
    if stringNum > 35:
        file_dir = "..." + file_dir[-25:]
    return file_dir


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.openFileNameLabel = QLabel()
        self.saveFolderLabel = QLabel()
        self.fileDir = None
        self.saveDir = None
        self.ui.Browser.clicked.connect(self.setOpenFileName)
        self.ui.Browser_2.clicked.connect(self.setSaveFolder)
        self.ui.Play.clicked.connect(self.startVideo)
        self.ui.Stop.clicked.connect(self.stopVideo)
        self.ui.Line_left.clicked.connect(self.setLeftLane)
        self.ui.Line_Center.clicked.connect(self.setCenterLane)
        self.ui.Line_Right.clicked.connect(self.setRightLane)
        self.ui.Line_left_2.clicked.connect(self.setLeftDirect)
        self.ui.Line_Center_2.clicked.connect(self.setCenterDirect)
        self.ui.Line_Right_2.clicked.connect(self.setRightDirect)
        self.ui.Square.clicked.connect(self.setRect)
        self.ui.Delete.clicked.connect(self.GoBack)
        self.ui.Save_video.clicked.connect(self.exportVideo)
        self.ui.Table.itemDoubleClicked.connect(self.openImage)
        self.ui.Table.setColumnWidth(0, 300)
        self.ui.Table.setColumnWidth(1, 100)
        self.ui.Close.clicked.connect(self.close)
        self.ui.Minimun.clicked.connect(self.showMinimized)
        self.capture_thread = None
        self.saving_thread = None
        self.stop = True
        self.begin = QPoint()
        self.end = QPoint()
        self.previous = {}
        self.current = {}
        self.counter = {}
        self.mapping = {}
        self.violation = []
        self.k_counter = None
        self.t_counter1 = []
        self.t_counter2 = []
        self.t_counter3 = []
        self.t_counter4 = []
        self.t_counter5 = []
        self.t_counter6 = []
        self.v_counter = []
        self.w_counter = []
        self.current_time = 0
        self.screen = self.ui.Camera_view.frameSize().height()
        self.ui.Camera_view = CameraView(self.ui.Camera_view)
        self.ui.Draw_line = DrawObject(self.ui.Draw_line)

    def setOpenFileName(self):
        self.fileDir, _ = QFileDialog.getOpenFileName(
            self,
            "Open video", self.openFileNameLabel.text(),
            "Videos (*.mp4)"
        )
        if self.fileDir:
            fileName = shortDir(self.fileDir)
            self.ui.Filename.setText(fileName)
            retval, img = getFrame(self.fileDir, 5)
            qImg = QtImage(self.screen, img)
            self.ui.Draw_line.setImage(qImg)
            self.saveDir = os.path.dirname(self.fileDir)
            self.ui.Filename_2.setText(self.saveDir)
            self.showSaveDir()

    def setSaveFolder(self):
        self.saveDir = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if self.saveDir:
            self.showSaveDir()

    def showSaveDir(self):
        folderName = self.saveDir
        folderName = shortDir(folderName)
        self.ui.Filename_2.setText(folderName)
        Saving_info = "Images will be saved as: <font color='green'> {} </font> <br>" \
                      "Videos will be saved as: <font color='green'> {} </font>".format(folderName + "/Images",
                                                                                        folderName + "/Videos")
        self.ui.Result_dir.setText(Saving_info)

    def setLeftLane(self):
        self.ui.Draw_line.setMode("lane", "left")

    def setCenterLane(self):
        self.ui.Draw_line.setMode("lane", "center")

    def setRightLane(self):
        self.ui.Draw_line.setMode("lane", "right")

    def setLeftDirect(self):
        self.ui.Draw_line.setMode("direct", "left")

    def setCenterDirect(self):
        self.ui.Draw_line.setMode("direct", "center")

    def setRightDirect(self):
        self.ui.Draw_line.setMode("direct", "right")

    def setRect(self):
        self.ui.Draw_line.setMode("rect", None)

    def GoBack(self):
        self.ui.Draw_line.goBack()

    def startVideo(self):
        if self.fileDir:
            self.stop = False
            images_path = os.path.join(self.saveDir, "Images")
            videos_path = os.path.join(self.saveDir, "Videos")
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            if not os.path.exists(videos_path):
                os.makedirs(videos_path)
            self.capture_thread = threading.Thread(target=self.update)
            self.capture_thread.start()

    def stopVideo(self):
        self.stop = True

    def openImage(self, index):
        keys = list(self.mapping.keys())
        file = 'Images/{}'.format(keys[index.row()])
        file_dir = os.path.join(self.saveDir, file)
        pixmap = QPixmap(file_dir)
        pixmap = pixmap.scaledToWidth(551)
        self.ui.Preview.setPixmap(pixmap)
        self.ui.Preview_name.setText(file)
        self.ui.Preview.show()

    def exportVideo(self):
        self.saving_thread = threading.Thread(target=self.savingVideo)
        self.saving_thread.start()

    def savingVideo(self):
        index = (self.ui.Table.selectionModel().currentIndex())
        keys = list(self.mapping.keys())
        file = 'Videos/{}.mp4'.format(keys[index.row()])
        file_dir = os.path.join(self.saveDir, file)
        clip = VideoFileClip(self.fileDir)
        start, end = self.mapping[keys[index.row()]]
        clip = clip.subclip(start, end)
        clip.write_videofile(file_dir, audio=False, verbose=False, logger=None)

    def updateTable(self, file_name):
        if self.current_time - 2 > 0:
            start = self.current_time - 2
        else:
            start = 0
        end = self.current_time + 2
        self.mapping[file_name] = [start, end]
        num_cols = 1
        num_rows = len(self.mapping.keys())
        self.ui.Table.setColumnCount(num_cols)
        self.ui.Table.setRowCount(num_rows)
        idx = 0
        for key, value in self.mapping.items():
            self.ui.Table.setItem(idx, 0, QTableWidgetItem('{}.jpg'.format(key)))
            idx += 1

    def updateWrongLane(self, img, x0, y0, violation, track_id, time_stamp):
        self.w_counter.append(track_id)
        self.ui.Vcounter_2.setText(str(len(list(set(self.w_counter)))))
        cv2.rectangle(img, (x0, y0 - 10), (x0 + 10, y0), (255, 0, 0), -1)
        file_name = "{}_{}".format(violation, time_stamp)
        if self.ui.SaveImage.isChecked():
            cv2.imwrite("{}/Images/{}.jpg"
                        .format(self.saveDir, file_name), img)
            self.ui.Violation_name.setText(file_name)
            if self.ui.SaveVideo.isChecked():
                self.updateTable(file_name)

    def updateCrossLight(self, img, x0, y0, track_id, time_stamp):
        self.v_counter.append(track_id)
        self.ui.Vcounter.setText(str(len(list(set(self.v_counter)))))
        cv2.rectangle(img, (x0, y0 - 10), (x0 + 10, y0), (0, 0, 255), -1)
        file_name = "vuot_den_{}".format(time_stamp)
        if self.ui.SaveImage.isChecked():
            cv2.imwrite("{}/Images/{}.jpg"
                        .format(self.saveDir, file_name), img)
            self.ui.Violation_name.setText(file_name)
            if self.ui.SaveVideo.isChecked():
                self.updateTable(file_name)


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

    def Checking(self):
        cap = cv2.VideoCapture(self.fileDir)
        myFrameNumber = 5
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
        while myFrameNumber < totalFrames:
            retval, img = cap.read()
            img_height, img_width, img_colors = img.shape
            scale = self.screen / img_height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            wd02 = QtImage(self.screen, img)
            self.ui.Export_view.setImage(wd02)

start = time.time()
app = QApplication(sys.argv)
end = time.time()
print('App {:.5f} seconds'.format(end - start))
w = MyWindow(None)
end = time.time()
print('MyWindow {:.5f} seconds'.format(end - start))
w.windowTitle()
end = time.time()
print('Title {:.5f} seconds'.format(end - start))
w.show()
end = time.time()
print('Show {:.5f} seconds'.format(end - start))
app.exec_()
end = time.time()
print('xxx {:.5f} seconds'.format(end - start))