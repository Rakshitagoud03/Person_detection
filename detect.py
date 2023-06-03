import math   ##inbuilt math function
import argparse
import os   ## accessing the local path
import sys     ## access the file or commands for file
from pathlib import Path      ## take the path of source
import time      ## hold the compiler executions using time.sleep(0-N)    
import cv2       ## image handling put text or draw lines , resizing , reading in color or gray
import torch     ## feature extracting the class detailed
import torch.backends.cudnn as cudnn   ## graphics cuda or cudnn  or system gpu
from models.common import DetectMultiBackend  ##  loading the object features from the tained weighths
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams        ## loading the formats from the utils
from utils.general import (check_file, check_img_size, check_imshow,increment_path, non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh) ##  general utililz
from utils.plots import Annotator, colors    ## annotations or bbox and 
from utils.torch_utils import select_device   ## choosing device

@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='0',  # file/dir/URL/glob, 0 for webcam , 1  for  usb cam
        data='data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp   16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
            
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
       
        # Process predictions
        for i, det in enumerate(pred):  # per image
            if webcam:  
                im0, frame = im0s[i].copy(), dataset.count
                #s += f'{i}: '
            else:
                im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain wh wh
            #imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                height, width, _ = im0.shape
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh    ## detecion quadinates
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        
                    c = int(cls)  # integer class
                    x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    gridx1 = int(width/2)
                    point = int((w-x)/2)+x

                    if names[c]=="person":
                        #Send('L')
                        
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                   
                    
        # Stream results
        im0 = annotator.result()
        #if view_img:
        cv2.imshow('Object Detection', im0)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
            break

if __name__ == "__main__":
    run()
