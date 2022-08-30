import os
import os.path
import glob
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import shutil
import random

#==================parameters=================
save_path = 'CMND_CCCD'
data_dir = "/data/data_HDD/convert_data/CMND_CCCD_19_10_2021/"
split_data = 0.8
label_dict={"CMND":0,"CCCD":1}
#========================================

train_folder = os.path.join(save_path,"train")
val_folder = os.path.join(save_path,"val")
def create_folder(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.mkdir(path)
    os.mkdir(train_folder)
    os.mkdir(val_folder)
create_folder(save_path)



class Customdata(data.Dataset):
    def __init__(self, dataroot, preproc=None):
        self.preproc = preproc
        self.list_images=glob.glob(os.path.join(dataroot,"*.jpg"))
        self.list_anotations=[x[:-3]+"json" for x in self.list_images]
        self.err=open("err.txt","w+")

    def __len__(self):
        return len(self.list_images)

    def get_labels(self,path_annotation):
        try:
            labels=[]
            with open(path_annotation) as json_file:
                data = json.load(json_file)
                width=data['imageWidth']
                height=data['imageHeight']
                for shape in data['shapes']:
                    lb=label_dict[shape['label']]
                    points=shape["points"]
                    (x1,y1),(x2,y2),(x3,y3),(x4,y4)=points
                
                    xmin=min(x1,x2,x3,x4)
                    ymin=min(y1,y2,y3,y4)
                    xmax=max(x1,x2,x3,x4)
                    ymax=max(y1,y2,y3,y4)
                    # if(y4>y2):
                        # x4,y4,x2,y2=x2,y2,x4,y4
                    labels.append([xmin,ymin,xmax,ymax,x1,y1,x2,y2,x4,y4,x3,y3,lb])
            return labels # xmin ymin xmax ymax : bounding box , (x1,y1) (x2,y2) (x3,y3) (x4,y4)  : top left,top right,bottom right , bottom left
        except:
            self.err.write(path_annotation+"\n")
            return []

    def __getitem__(self, index):
        img = cv2.imread(self.list_images[index])
        height, width, _ = img.shape

        labels = self.get_labels(self.list_anotations[index])
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # xmin
            annotation[0, 1] = label[1]  # ymin
            annotation[0, 2] = label[2]   # xmax
            annotation[0, 3] = label[3]   # ymax

            # landmarks
            annotation[0, 4] = label[4]    #x1
            annotation[0, 5] = label[5]    #y1
            annotation[0, 6] = label[6]    # x2
            annotation[0, 7] = label[7]    # y2

            annotation[0, 8] = (label[0]+label[2])/2  # x center
            annotation[0, 9] = (label[1]+label[3])/2 # y_center

            
            annotation[0, 10] = label[8]  # x3
            annotation[0, 11] = label[9]   # y3
            annotation[0, 12] = label[10]  # x4
            annotation[0, 13] = label[11]  # y4

            
            if (annotation[0, 4]<0):
                annotation[0, 14] = label[12]
            else:
                annotation[0, 14] = label[12]

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)



aa=Customdata(data_dir)
for i in range(len(aa.list_images)):
    print(i, aa.list_images[i])
    img = cv2.imread(aa.list_images[i])
    base_img = os.path.basename(aa.list_images[i])
    base_txt = os.path.basename(aa.list_images[i])[:-4] +".txt"
    if(random.randint(0,100)>split_data*100):
        save_img_path = os.path.join(val_folder, base_img)
        save_txt_path = os.path.join(val_folder, base_txt)
    else:
        save_img_path = os.path.join(train_folder, base_img)
        save_txt_path = os.path.join(train_folder, base_txt)
    with open(save_txt_path, "w") as f:
        height, width, _ = img.shape
        labels = aa.get_labels(aa.list_anotations[i])
        print(labels)
        annotations = np.zeros((0, 14))
        if len(labels) == 0:
            continue
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 14))
            # bbox
          
            #if (label[2] -label[0]) < 8 or (label[3] - label[1]) < 8:
            #    img[int(label[1]):int(label[3]), int(label[0]):int(label[2])] = 127
            #    continue
            # landmarks
            annotation[0, 4] = label[4] / width  # l0_x
            annotation[0, 5] = label[5] / height  # l0_y
            annotation[0, 6] = label[6] / width  # l1_x
            annotation[0, 7] = label[7]  / height # l1_y

            annotation[0, 8] = (label[0]+label[2])/(2*width)  # l2_x
            annotation[0, 9] = (label[1]+label[3])/(2*height)  # l2_y

            annotation[0, 10] = label[8] / width  # l3_x
            annotation[0, 11] = label[9] / height  # l3_y
            annotation[0, 12] = label[10] / width  # l4_x
            annotation[0, 13] = label[11] / height  # l4_y


            label[0] = max(0, label[0])
            label[1] = max(0, label[1])
            label[2] = min(width -  1, label[2]-label[0])
            label[3] = min(height - 1, label[3]-label[1])
            annotation[0, 0] = (label[0] + label[2] / 2) / width  # cx
            annotation[0, 1] = (label[1] + label[3] / 2) / height  # cy
            annotation[0, 2] = label[2] / width  # w
            annotation[0, 3] = label[3] / height  # h


            str_label=str(label[12])+" "
            for i in range(len(annotation[0])):
                str_label =str_label+" "+str(annotation[0][i])
            str_label = str_label.replace('[', '').replace(']', '')
            str_label = str_label.replace(',', '') + '\n'
            f.write(str_label)
    cv2.imwrite(save_img_path, img)

