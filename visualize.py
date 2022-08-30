import numpy as np





import cv2

def draw_rectangle(img, box, color, thickness: int = 3):
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(
        box[2]), int(box[3])), color, thickness)
    return img


def draw_text(img, text, above_box, color=(0, 10, 255)):
    tl_pt = (int(above_box[0]), int(above_box[1]) - 7)
    cv2.putText(img, text, tl_pt,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=color)
    return img


def draw_trace(image, trace, color):
    for pos in trace:
        image = cv2.circle(image, (int(pos[0]+pos[2]/2),int(pos[1]+pos[3]/2)), 1, color, 3)
    return image


def draw_track(img, track,text, draw: bool = True, fallback_color=(255, 255, 255)):
    color = (0,255,122)
    x0 = int(track[0])
    y0 = int(track[1])
    x1 = int(track[2])
    y1 = int(track[3])
    box=[x0,y0,x1,y1]
    if(draw ):
        img = draw_rectangle(img, box, color=color, thickness=5)
    img = draw_trace(img, track[6], color)
    
    img = draw_text(img, text, above_box=box)
    return img


def draw_detection(img, detection):
    img = draw_rectangle(img, detection.box, color=(0, 220, 0), thickness=1)
    return img


def draw_detection_box(img, detection):
    img = draw_rectangle(img, detection, color=(0, 220, 0), thickness=1)
    return img


def image_generator(*args, **kwargs):

    def _empty_canvas(canvas_size=(CANVAS_SIZE, CANVAS_SIZE, 3)):
        img = np.ones(canvas_size, dtype=np.uint8) * 30
        return img

    data_gen = data_generator(*args, **kwargs)
    for dets_gt, dets_pred in data_gen:
        img = _empty_canvas()

        # overlay actor shapes
        for det_gt in dets_gt:
            xmin, ymin, xmax, ymax = det_gt.box
            feature = det_gt.feature
            for channel in range(3):
                img[int(ymin):int(ymax), int(xmin):int(
                    xmax), channel] = feature[channel]

        yield img, dets_gt, dets_pred


if __name__ == "__main__":
    for img, dets_gt, dets_pred in image_generator(
            num_steps=1000, num_objects=10):

        for det_gt, det_pred in zip(dets_gt, dets_pred):
            img = draw_rectangle(img, det_gt.box, color=det_gt.feature)

            if det_pred.box is not None:
                img = draw_rectangle(
                    img, det_pred.box, color=det_pred.feature, thickness=1)

        cv2.imshow('preview', img)
        c = cv2.waitKey(33)
        if c == ord('q'):
            break
