import glob
import json
import math
import os
import time

import torch

import numpy as np
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import (Profile, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device

os.environ['PATH'] = 'C:/tools/vips_dev/bin' + ';' + os.environ['PATH']

import pyvips

device = select_device('0')
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())


def load_model(model_path, data_path):
    model = DetectMultiBackend(model_path, device, False, data_path, False)
    # stride, names, pt = model.stride, model.names, model.pt
    # print(stride, names, pt)
    return model


def prepare_image(image, img_size=640):
    img = letterbox(image.copy(), new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img, image


def load_image(im0, img_size, stride=32, auto=True):
    im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    return im, im0


def get_abs_position(xywh, w, h):
    x1 = int((xywh[0] - xywh[2] / 2) * w)
    y1 = int((xywh[1] - xywh[3] / 2) * h)
    x2 = int((xywh[0] + xywh[2] / 2) * w)
    y2 = int((xywh[1] + xywh[3] / 2) * h)
    return x1, y1, x2, y2


def detect(model, img, imgsz=(1024, 1024)):
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    im, im0 = load_image(img, img_size=1024)

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    with dt[2]:
        pred = non_max_suppression(pred, 0.94, 0.45, None, False, max_det=1000)
    cls_result = []
    detection_result = []
    detection_score = []

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                print(conf)
                c = int(cls)
                _x0, _y0, _x1, _y1 = [int(x) for x in xyxy]
                cls_result.append(str(c))
                detection_result.append([str(c), _x0, _y0, _x1, _y1])
                detection_score.append(float(conf))
    return cls_result, detection_result, detection_score, img


if __name__ == '__main__':
    md = load_model('best.pt', 'mydata.yaml')
    tifs = glob.glob('detection/*.tif')
    out_path = './out/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for tif in tifs:
        s = time.time()
        target = pyvips.Image.new_from_file(tif)
        fileout = f"{out_path}/{tif.replace('.tif', '.json')}"
        if not os.path.exists(os.path.dirname(fileout)):
            os.makedirs(os.path.dirname(fileout))
        col, y = divmod(target.width, 1024)
        _x_ignore = y // 2
        row, y_1 = divmod(target.height, 1024)
        _y_ignore = y_1 // 2
        result = {
            'id': int(os.path.basename(fileout)[:-5]),
            'slide_label': 1,
        }
        annotations = []
        h_index = 1
        for i in range(math.ceil(row)):
            for j in range(math.ceil(col)):
                _x = j * 1024
                _y = i * 1024

                if _x + 1024 > target.width:
                    w = target.width - _x
                else:
                    w = 1024
                if _y + 1024 > target.height:
                    h = target.height - _y
                else:
                    h = 1024

                start_x = _x + _x_ignore
                start_y = _y + _y_ignore
                try:
                    t = target.extract_area(start_x, start_y, w, h)
                except Exception as e:
                    continue
                tim = np.ndarray(buffer=t.write_to_memory(), dtype=np.uint8,
                                 shape=[t.height, t.width, t.bands])
                t1 = time.time()
                cls, bboxes, score, im = detect(md, tim)
                tmp = {
                    'id': h_index,
                    'x': start_x,
                    'y': start_y,
                    'width': 1024,
                    'height': 1024
                }
                index = 1
                box = []
                for bbox in bboxes:
                    label, x1, y1, x2, y2 = bbox
                    t = {
                        'id': index,
                        'label': md.names[int(label)],
                        'x1': start_x + x1,
                        'y1': start_y + y1,
                        'x2': start_x + x2,
                        'y2': start_y + y2
                    }
                    box.append(t)
                    index += 1
                tmp['bboxes'] = box
                if len(box) == 0:
                    continue
                annotations.append(tmp)
                h_index += 1
        result['annotations'] = annotations
        with open(fileout, 'w') as w:
            w.write(json.dumps(result))
        print(time.time() - s)
