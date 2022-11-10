
### importing required libraries
import torch
import cv2
import numpy as np
#test
from .src.data_utils import convert2Square
from skimage.filters import threshold_local
import imutils
from skimage import measure
from .src.char_classification.model import CNN_Model

##### DEFINING GLOBAL VARIABLE

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'





### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    #test
    # results.show()
    print( results.xyxyn[0])
    print(results.xyxyn[0][:, -1])
    print(results.xyxyn[0][:, :-1])
    #test
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            # cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

            # coords = (x1,y1,x2,y2)
            crop_image = frame[y1:y2,x1:x2]
            # read_text(crop_image)
            plate_segment = segmentation(crop_image)
            plate_charRecog = charRecog(plate_segment)
            plate_num = format(plate_charRecog)
            print(plate_num)

            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 2)

            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])




    return frame



#### ---------------------------- function to recognize license plate --------------------------------------

def segmentation(LpRegion):
    segment_result=[]
    #apply thresh to extracted license plate
    LpRegion = cv2.cvtColor(LpRegion, cv2.COLOR_RGB2HSV)
    V = cv2.split(LpRegion)[2]

    # adaptive threshold
    T = threshold_local(V, 15, offset = 10, method = "gaussian")
    thresh = (V > T).astype("uint8") * 255
    
    #convert black pixel of digital to white pixel 
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=400)
    thresh = cv2.medianBlur(thresh,5)

    # connected components analysis
    labels = measure.label(thresh, connectivity=2, background=0)

    # loop over the unique components
    for label in np.unique(labels):
        # if this is background label, then ignore it.
        if label == 0:
            continue
        # init mask  to store the location  of the character  candidates
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

    # find contours from mask 
    contours, hierarchy =  cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)

        #rules to determine characters
        aspectRatio = w/ float(h)
        solidity = cv2.contourArea(contour)/ float(w*h)
        heightRatio = h/ float(LpRegion.shape[0])
        if 0.1 < aspectRatio < 1.0 and  solidity > 0.1 and 0.35 < heightRatio < 2.0:
            # extract chararters
            candidate = np.array(mask[y:y+h, x:x+w])
            square_candidate = convert2Square(candidate)
            square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
            square_candidate = square_candidate.shape((28,28,1))
            segment_result.append(square_candidate, (y,x))
    return segment_result
# function to recognize license plate numbers using  OCR
def charRecog(candidates):
    recogChar = CNN_Model(False).model
    recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
    characters = []
    coordinates = []
    for char, coordinate in candidates:
        characters.append(char)
        coordinates.append(coordinate)
    
    characters = np.array(characters)
    result = recogChar.predict_on_batch(characters)
    result_idx = np.argmax(result, axis=1)
    candidates = []
    for i in range(len(result_idx)):
        if result_idx[i] == 31: # if this is background or noise, then ignore it
            continue
        candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))
    return candidates

def format(candidates):
    first_line = []
    second_line = []

    for candidate, coordinate in candidates:
        if candidates[0][1][0] + 40 > coordinate[0]:
            first_line.append((candidate, coordinate[1]))
        else:
            second_line.append((candidate, coordinate[1]))

    def take_second(s):
        return s[1]

    first_line = sorted(first_line, key=take_second)
    second_line = sorted(second_line, key=take_second)

    if len(second_line) == 0:  # if license plate has 1 line
        license_plate = "".join([str(ele[0]) for ele in first_line])
    else:   # if license plate has 2 lines
        license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

    return license_plate





### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally

    classes = model.names ### class names in string format




    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # cv2.imshow("frames1", frame)
        frame = plot_boxes(results, frame,classes = classes)
        # cv2.imshow("frames2", frame)


        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")

                cv2.imwrite(f"{img_out_name}",frame) ## if you want to save he output result.

                break

    ### --------------- for detection on video --------------------
    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret  and frame_no %1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                # cv2.imshow("",results)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


                frame= plot_boxes(results, frame,classes = classes)

                cv2.imshow("vid_out", frame)
            
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1
        
        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


# ### for custom video

# main(vid_path="test.MOV",vid_out="result/mp4") #### for webcam

main(img_path="test_images/5.jpg") ## for image
            

