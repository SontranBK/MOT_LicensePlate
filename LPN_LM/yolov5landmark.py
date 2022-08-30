from LPN_LM.models.yolo import Model
from LPN_LM.alignmet_four_point import Alignment
from LPN_LM.utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh
from LPN_LM.utils.datasets import letterbox
from LPN_LM.models.experimental import attempt_load
import glob
import copy
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import cv2
from pathlib import Path
import time
from re import M
import os
import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
path_cur = os.path.dirname(os.path.abspath(__file__))


class Yolov5Landmark:
    def __init__(self, img_size=640, weight_path="weights/lpn_best_5s_statedict.pt", config_path="models/yolov5s.yaml"):
        self.img_size = img_size
        self.weight_path = os.path.join(path_cur, weight_path)
        self.conf_thres = 0.3
        self.iou_thres = 0.5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = os.path.join(path_cur, config_path)
        self.alignment = Alignment()
        self.load_model()

    def load_model(self):
        # load default
        # self.model = attempt_load(
        # "LPN_LM/weights/lpn_best_5m.pt", map_location=self.device)  # load FP32 model
        # torch.save(self.model.state_dict(),"LPN_LM/weights/lpn_best_5m_statedict.pt")

        self.model = Model(self.config_path).to(self.device).fuse()
        self.model.load_state_dict(torch.load(
            self.weight_path))
        self.model.eval()

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:  # calculate from img0_shape
            # gain  = old / new
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    def show_results(self, img, xywh, conf, landmarks, class_num):
        h, w, c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0),
                      thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (0, 255, 255)]

        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

        tf = max(tl - 1, 1)  # font thickness
        label = str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def detect(self, orgimg):
        img_aligneds = []
        boxes = []
        img0 = copy.deepcopy(orgimg)
        # img_drawed = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)),
                              interpolation=interp)
        imgsz = check_img_size(
            self.img_size, s=self.model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(
                self.device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(
                self.device)  # normalization gain landmarks
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], orgimg.shape).round().cpu()
                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                det[:, 5:15] = self.scale_coords_landmarks(
                    img.shape[2:], det[:, 5:15], orgimg.shape).round()

                for j in range(det.size()[0]):

                    # xywh = (xyxy2xywh(det[j, :4].view(
                    #     1, 4)) / gn).view(-1).tolist()
                    xyxy = [int(x) for x in det[j, :4].tolist()]
                    
                    conf = det[j, 4].detach().cpu().numpy()
                    land = (det[j, 5:15].view(1, 10) /
                            gn_lks).view(-1).tolist()
                    img_aligneds.append(self.alignment.align(orgimg, [(
                        land[0]*w0, land[1]*h0), (land[2]*w0, land[3]*h0), (land[8]*w0, land[9]*h0), (land[6]*w0, land[7]*h0)]))
                    class_num = det[j, 15].detach().cpu().numpy()
                    boxes.append(xyxy)
                    # img_drawed = self.show_results(
                    #     orgimg, xywh, conf, land, class_num)

        return boxes,img_aligneds


if __name__ == '__main__':
    x = Yolov5Landmark()
    img = cv2.imread("4.jpg")
    for i in range(1):
        t1 = time.time()
        img_aligneds, img_drawed = x.detect(img)
        print("number lp ", len(img_aligneds))
        cv2.imshow("image origin  ", img)
        cv2.imshow("img_drawed  ", img_drawed)
        for i in range(len(img_aligneds)):
            cv2.imshow("image aligned "+str(i), img_aligneds[i])
        cv2.waitKey(0)
        print(" time ", time.time()-t1)
