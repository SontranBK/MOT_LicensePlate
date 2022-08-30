from visualize import draw_track
import pickle
import sys
import cv2
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
sys.path.insert(0, 'Detection')
sys.path.insert(0, 'Tracking')
from PIL import Image, ImageDraw
from charlp_detection.detect import CHARLP
from Tracking.bytetrack import BYTETracker
from Detection.yolov5.detect import Yolov5
import numpy as np
FINAL_LINE_COLOR = (255, 0, 255)


class Counter:
    def __init__(self, data_path, url='town.avi'):
        self.detector = Yolov5(
            list_objects=["bicycle", "car", "truck", "motorcycle", "bus"])
        self.tracker = BYTETracker(track_thresh=0.5, track_buffer=30,
                                   match_thresh=0.8, min_box_area=10, frame_rate=30)
        ##License Plate
        
        self.charlp = CHARLP()
        from LPN_LM.yolov5landmark import Yolov5Landmark
        self.lpn_lm = Yolov5Landmark()
        
        self.url = url
        self.cam = cv2.VideoCapture(url)
        _, frame = self.cam.read()
        self.shape = frame.shape[:2]
        self.mark_in = {}
        self.mask_out = {}
        self.height, self.width = frame.shape[:2]
        self.counter_on = {"bicycle": 0, "car": 0,
                           "truck": 0, "motorcycle": 0, "bus": 0}
        self.counter_off = {"bicycle": 0, "car": 0,
                            "truck": 0, "motorcycle": 0, "bus": 0}
        self.map = {1: "bicycle", 2: "car",
                    3: "motorcycle", 5: "truck", 7: "truck"}
        self.data_show = {}
        self.lpn_res={}
        self.create_mask(data_path)
        self.frame_count=0
        self.nframe_lp=2
        self.min_size_lpn=15

    def load_polygon(self, data_path):
        data = pickle.load(open(data_path, "rb"))
        ratio_x = self.shape[1]/data["shape"][1]
        ratio_y = self.shape[0]/data["shape"][0]
        self.polygon = [(int(x[0]*ratio_x), int(x[1]*ratio_y))
                        for x in data["polygon"]]
        self.polygon.append(self.polygon[0])

    def create_mask(self, data_path):
        self.load_polygon(data_path)
        img = Image.new('L', (self.width, self.height), 0)
        ImageDraw.Draw(img).polygon(self.polygon, outline=1, fill=1)
        self.mask = np.array(img)

    def set_mask(self, polygon):
        self.polygon = polygon
        self.create_mask()
    
     
    
    def get_lpn(self,frame,id_tracking,box):
    
        if(id_tracking not in self.lpn_res or self.frame_count - self.lpn_res[id_tracking]["frame_id"]>self.nframe_lp ):
            img_obj = frame[box[1]:box[3],box[0]:box[2]].copy()
            try:
                boxes,img_aligneds=self.lpn_lm.detect(img_obj)
            except:
                boxes=[]
            if(len(boxes)==0) : return "",0
            s=0
            index=0
            for i in range(len(boxes)):
                s_cur=(boxes[i][2]-boxes[i][0])* (boxes[i][3]-boxes[i][1])
                if(boxes[i][2]-boxes[i][0]<self.min_size_lpn or boxes[i][3]-boxes[i][1]<self.min_size_lpn ): continue
                if(s_cur>s):
                    s=s_cur
                    index=i
            if((id_tracking not in self.lpn_res) or  (s>self.lpn_res[id_tracking]["s"])):
                if(s<self.min_size_lpn): return "",0
                box_detects,text = self.charlp.detect(img_aligneds[index])
                return text,s
            else:
                return self.lpn_res[id_tracking]["text"],self.lpn_res[id_tracking]["s"]
        else:
            return -1,-1
                

    def process_trackers(self, frame, tracks):
        self.frame_count+=1
        for data in tracks:
            draw = False
            id = data[4]
            label = data[5][0]
            trace = data[6]
            x0 = int(data[0])
            y0 = int(data[1])
            x1 = int(data[2])
            y1 = int(data[3])
            box=[x0,y0,x1,y1]
            text,s = self.get_lpn(frame,id,box)
            if(text!=-1):
                self.lpn_res[id]={"text":text,"s":s,"frame_id":self.frame_count}
            else:
                text = self.lpn_res[id]["text"]
            
            
            if(len(trace) > 1):
                pos = trace[-1]
                x1, y1 = (int(pos[0]+pos[2]/2), int(pos[1]+pos[3]/2))
                pos = trace[-2]
                x2, y2 = (int(pos[0]+pos[2]/2), int(pos[1]+pos[3]/2))

                if(self.mask[y1][x1] == False and self.mask[y2][x2] == True):  # in
                    if(id not in self.mark_in):
                        self.mark_in[id] = 1
                        self.counter_on[self.map[label]] += 1
                        draw = True
                    if(id in self.mask_out):
                        self.counter_off[self.map[label]] -= 1
                        self.mask_out.pop(id)
                elif(self.mask[y1][x1] == True and self.mask[y2][x2] == False):  # out
                    if(id in self.mark_in):
                        self.counter_on[self.map[label]] -= 1
                        self.mark_in.pop(id)
                    if(id not in self.mask_out):
                        self.counter_off[self.map[label]] += 1
                        self.mask_out[id] = 1
                        draw = True
            draw_track(frame, data, draw=draw,text=text)

    def put_res(self, frame):
        color = (0, 0, 255)
        d = 0
        h,w=frame.shape[:2]
        total_in=0
        total_out=0
        size_skip=40
        type_count =0
        for key in self.counter_on.keys():
            frame = cv2.putText(frame, '{} out : '.format(key)+str(self.counter_on[key]), (w-200, d+size_skip//2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2, cv2.LINE_AA)
            total_out+=self.counter_on[key]
            frame = cv2.putText(frame, '{} in : '.format(key)+str(self.counter_off[key]), (w-200, d+size_skip), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2, cv2.LINE_AA)
            total_in+=self.counter_off[key]
            d += size_skip
            type_count+=1
            
        str_total ="TOTAL : IN {} , OUT {}".format(total_in,total_out)
        frame = cv2.putText(frame, str_total, (w-200, size_skip*(type_count+1)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,0,0), 2, cv2.LINE_AA)
        
        return frame

    def Detect(self, detector, frame):
        box_detects, classes, confs = detector.detect(frame.copy())
        return np.array(box_detects).astype(int), np.array(confs), np.array(classes)

    def run(self):
        video = self.cam
        frame_num = 0
        ret, frame = video.read()
        frame_ori = frame.copy()
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('output.avi',fourcc, 18, (width+200,height))
        while 1:
            # try:
            # print('--------------------------------')
            # detection = []
            ret, frame = video.read()
            if(frame is None ):
                print(" Camera stop ")
                break
            
            frame_num += 1
            if frame_num % 1 == 0:  # skip_frame
                # start = time()

                box_detects, scores, classes = self.Detect(
                    self.detector, frame)

                data_track = self.tracker.update(
                    box_detects, scores, classes)

                self.process_trackers(frame, data_track)

                # print("time : ",time()-start)
                #Draw ouput
                frame = cv2.polylines(frame, np.array(
                    [self.polygon]), False, FINAL_LINE_COLOR, 1)
                
                frame = cv2.copyMakeBorder(
                    frame,
                    top=0,
                    bottom=0,
                    left=0,
                    right=200,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )
                frame = self.put_res(frame)
                out_video.write(frame)

                cv2.imshow('frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            # except:
            #     pass
        # out_video.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    X = Counter("data/polygon.pickle", url="/home/haobk/video.mp4")
    X.run()
