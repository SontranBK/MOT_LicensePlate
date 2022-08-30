import numpy as np
from charlp_detection.models.yolo import Model
from charlp_detection.utils.general import non_max_suppression, scale_coords
from charlp_detection.models.experimental import attempt_load
from charlp_detection.utils.datasets import letterbox
import torch
import cv2
import glob
import os
import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
path_cur = os.path.dirname(os.path.abspath(__file__))


class CHARLP():
    def __init__(self):
        self.device = torch.device("cuda")
        self.path_model = os.path.join(path_cur, "weights/lpn_best_5m_statedict.pt")
        # self.model=attempt_load(self.path_model, map_location="cuda")
        self.config_path = os.path.join(path_cur, "models/char_lp.yaml")
        self.load_model()
        self.names =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']

        self.img_size = 320
        self.conf_thres = 0.25
        self.iou_thres = 0.25

    def load_model(self):
        # load default
        # self.model = attempt_load(
        # "charlp_detection/weights/lp_rec_yolov5.pt", map_location=self.device)  # load FP32 model
        # self.names = model.module.names if hasattr(self.model, 'module') else self.model.names
        # print(self.names)
        # torch.save(self.model.state_dict(),"charlp_detection/weights/lpn_best_5m_statedict.pt")

        self.model = Model(self.config_path).to(self.device).float()
        self.model.load_state_dict(torch.load(self.path_model))
        self.model.eval()

    def taken(self, elem):
        return elem['center'][1]

    def taken1(self, elem):
        return elem['center'][0]

    def Sort(self, res):
        res.sort(key=self.taken)
        dong1 = []
        try:
            dong1.append(res[0])
        except:
            dong1 = []
        dong2 = []
        for i in range(1, len(res)-1):
            height = res[i]['height']
            if((height/3+1) < res[i+1]['center'][1]-res[i]['center'][1]):
                dong1.append(res[i])
                for j in range(i+1, len(res)):
                    dong2.append(res[j])
                break
            else:
                dong1.append(res[i])
        if(dong2 == [] and len(dong1) > 0):
            dong1.append(res[-1])
        dong1.sort(key=self.taken1)
        dong2.sort(key=self.taken1)
        result = {"top": dong1, "bottom": dong2}
        # for x in dong1:
        #     result.append(x)
        # for x in dong2:
        #     result.append(x)
        return result

    def detect(self, im0s, draw=False, margin=0):
        img = letterbox(im0s.copy(), new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects = []
        ims = []
        # classes=[]
        confs = []
        cls_ids = []
        labels = []
        res = []
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=None, agnostic=None)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):

                    # print(conf)
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    top = c1[1]-margin
                    left = c1[0] - margin
                    right = c2[0] + margin
                    bottom = c2[1] + margin
                    # ims.append(im0s[top:bottom,left:right])
                    # labels.append(self.names[int(cls)])
                    box_detects.append([left, top, right, bottom])
                    res.append({'center': (int((left+right)/2), (top+bottom)/2),
                               'height': bottom-top, "lb": self.names[int(cls)]})
        result = self.Sort(res)
        text = ""
        for x in result['top']:
            text += x['lb']
        for x in result['bottom']:
            text += x['lb']

        if(draw):
            img = im0s
            font = cv2.FONT_HERSHEY_SIMPLEX
            for box, lb in zip(box_detects, classes):
                img = cv2.rectangle(
                    img, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]), (0, 255, 0), 3, 3)
                # img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)

        return box_detects, text  # bbox_xywh, cls_conf, cls_ids


if __name__ == '__main__':

    detector = YOLOV5()
    for path in glob.glob("data_LP_test/easy/*.jpg"):

        img = cv2.imread(path)

        boxes, ims, classes, img = detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX
        for box, im, lb in zip(boxes, ims, classes):
            print(lb)
            img = cv2.rectangle(
                img, (box[0], box[1]), (box[2]+box[0], box[3]+box[1]), (0, 255, 0), 3, 3)
            img = cv2.putText(
                img, lb, (box[0], box[1]), font, 2, (255, 0, 0), 1)
#         cv2.imshow("image",cv2.resize(img,(500,500)))
        cv2.waitKey(0)
