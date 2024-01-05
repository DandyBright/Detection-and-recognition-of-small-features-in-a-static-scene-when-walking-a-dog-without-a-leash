import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import sys
import json
sys.path.insert(1, '/project/train/src_repo/yolov7/')
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import set_logging
from utils.plots import plot_one_box
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from ultralytics.yolo.utils import ops
from ultralytics.nn.autobackend import AutoBackend
# ####### 参数设置
conf_thres = 0.24
iou_thres = 0.2
imgsz = 640
weights = "/project/train/src_repo/pretrain_model/yolov8n.pt"
weights_seg = "/project/train/src_repo/pretrain_model/yolov8n-seg.pt"
device = '0'
stride = 32
names = ['dog', 'person', 'cat']
names_seg = ['background', 'leash', 'dog']
def init():
    # Initialize
    global imgsz, device, stride
    set_logging()
    device = select_device('0')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = AutoBackend(weights, device=device, fp16=True)
    model_seg = AutoBackend(weights_seg, device=device, fp16=True)
    imgsz = check_imgsz(imgsz, stride=stride)  # check img_size
    model.eval()
    model_seg.eval()
    # model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    return model,model_seg

def process_image(model, input_image=None, args=None, **kwargs):
    # Padded resize
    if args is not None:
        cfg = json.loads(args)
    t0 = time.time()
    img0 = input_image
    img = letterbox(img0, new_shape=imgsz, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half()
    #     img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]
    t1 = time.time()
    print(t1-t0)
    with torch.no_grad():
        pred = model[0](img, augment=False)[0]
    t2 = time.time()
    print(t2-t1)
    
    # Apply NMS        
    pred = ops.non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    fake_result = {}
    fake_result["algorithm_data"] = {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
   }
    fake_result["model_data"] = {"objects": []}
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                fake_result["model_data"]['objects'].append({
                    "x":int(xyxy[0]),
                    "y":int(xyxy[1]),
                    "width":int(xyxy[2]-xyxy[0]),
                    "height":int(xyxy[3]-xyxy[1]),
                    "confidence":float(conf),
                    "name":names[int(cls)]
                    })

    if args is not None and 'mask_output_path' in cfg.keys() and cfg['mask_output_path']:
        fake_result["model_data"]["mask"] = cfg['mask_output_path']
        with torch.no_grad():
            pred_seg = model[1](img, augment=False)
        p = ops.non_max_suppression(pred_seg[0], conf_thres, iou_thres, agnostic=False, nm=32)
        proto = pred_seg[1][-1]
        for i, det in enumerate(p):  # detections per image
            if det is not None and len(det):
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                masks = ops.process_mask_native(proto[i], det[:, 6:], det[:, :4], img0.shape[:2])
                masks = masks.cpu().numpy().astype(int)
                for j in range(masks.shape[0]):
                    if int(det[j][5]) == 2:
                        masks[j][masks[j] == 1] = 2
                if masks.shape[0] == 2:
                    merged_mask = np.maximum.reduce([masks[0], masks[1]])
                    cv2.imwrite(cfg['mask_output_path'], merged_mask)
                else:
                    cv2.imwrite(cfg['mask_output_path'], masks[0])

    fake_result ["algorithm_data"]["target_info"]=[]
    return json.dumps(fake_result, indent = 4)

if __name__ == '__main__':
    from glob import glob
    # Test API
    image_names = glob('/home/data/2783/*.jpg')
    predictor = init()
    s = 0
    args = {"mask_output_path": "mask_result.png"}
    #args = {"mask_output_path": ''}
    args = json.dumps(args, indent = 4)
        
    for image_name in image_names:
        print(image_name)
        img = cv2.imread(image_name)
        t1 = time.time()
        res = process_image(predictor, img, args)
        print(res)
        t2 = time.time()
        s += t2 - t1
        break
    print(1/(s/100))