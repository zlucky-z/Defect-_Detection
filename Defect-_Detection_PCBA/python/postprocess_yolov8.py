# -*- coding: utf-8 -*-
"""
PCB缺陷检测 - NumPy后处理模块
优化版本，专门针对[1, 6, 8400]格式的模型输出
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional, Union

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    
    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes.
        eps (float): A small value to avoid division by zero. Default: 1e-7.
        
    Returns:
        iou (np.ndarray): An NxM numpy array containing the pairwise IoU values for every element in box1 and box2.
    """
    
    # Convert to numpy if needed
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    
    # Intersection area
    inter = np.maximum(0, np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)) * \
            np.maximum(0, np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1))
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1[:, None] * h1[:, None] + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    return iou

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    device = 'cpu'  # 我们使用numpy，所以device总是cpu
    mps = False      # 不使用MPS
    
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    prediction = np.array(prediction)
    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].astype(int) + 4] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(x, [4, 4 + nc], 1)
        
        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(np.float32), mask[i]), 1)
        else:  # best class only
            conf = cls.max(1, keepdims=True)
            j = cls.argmax(1).reshape(-1, 1).astype(np.float32)
            x = np.concatenate((box, conf, j, mask), 1)[conf.flatten() > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        
        # sort by confidence and remove excess boxes
        x = x[np.argsort(-x[:, 4])][:max_nms]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # Convert boxes to numpy array
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS
        i = nms_numpy(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def nms_numpy(boxes, scores, iou_threshold):
    """
    NumPy implementation of Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy arrays
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Get indices of sorted scores (descending order)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(sorted_indices) > 0:
        # Pick the box with highest score
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        iou = box_iou(current_box, remaining_boxes).flatten()
        
        # Keep boxes with IoU less than threshold
        keep_mask = iou < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return np.array(keep, dtype=int)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (numpy.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling and padding. If not provided, the ratio and pad will be
                           calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                        rescaling.

    Returns:
        boxes (numpy.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """
    Clips bounding boxes to image boundaries.

    Args:
        boxes (numpy.ndarray): bounding boxes, in format (x1, y1, x2, y2).
        shape (tuple): image shape, in format (height, width).

    Returns:
        None: modifies boxes in-place.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

class PostProcess:
    """优化的后处理类，专门处理PCB缺陷检测的[1, 6, 8400]格式输出"""
    
    def __init__(self, conf_thresh=0.3, nms_thresh=0.5, agnostic=False):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic = agnostic
        
    def __call__(self, predictions):
        """
        处理模型预测结果
        
        Args:
            predictions: 模型输出列表，每个元素为numpy数组
            
        Returns:
            处理后的检测结果列表
        """
        if not predictions:
            return [[]]
            
        # 假设只有一个批次
        prediction = predictions[0]
        
        # 检查输入格式
        if len(prediction.shape) == 3 and prediction.shape[1] == 6:
            # [1, 6, 8400] 格式 - 转换为标准格式
            prediction = self._convert_from_custom_format(prediction)
        elif len(prediction.shape) == 3 and prediction.shape[2] == 6:
            # [1, 8400, 6] 格式 - 已经是标准格式
            pass
        else:
            raise ValueError(f"不支持的预测格式: {prediction.shape}")
            
        # 应用NMS
        results = non_max_suppression(
            prediction,
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh,
            agnostic=self.agnostic
        )
        
        return results
    
    def _convert_from_custom_format(self, prediction):
        """
        将 [1, 6, 8400] 格式转换为标准的 [1, 8400, 6] 格式
        
        Args:
            prediction: shape为[1, 6, 8400]的numpy数组
            
        Returns:
            转换后的numpy数组，shape为[1, 8400, 6]
        """
        # 转置: [1, 6, 8400] -> [1, 8400, 6]
        converted = np.transpose(prediction, (0, 2, 1))
        
        return converted

print(f"LOADED FILE: {__file__}")
