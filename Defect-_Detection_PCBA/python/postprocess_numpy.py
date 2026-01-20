#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import numpy as np
# import cv2  # 注释掉cv2导入，因为代码中没有使用

print('LOADED FILE:', __file__)

class PostProcess:
    def __init__(self, conf_thresh=0.1, nms_thresh=0.5, agnostic=False, multi_label=True, max_det=1000, net_h=640, net_w=640):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.agnostic_nms = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.net_h = net_h
        self.net_w = net_w
        self.nms = pseudo_torch_nms()

        self.nl = 3
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, 1, -1, 1, 1, 2)
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def decode_for_3outputs(self, outputs):
        z = []
        for i, feat in enumerate(outputs):
            bs, _, ny, nx, nc = feat.shape
            if self.grid[i].shape[2:4] != feat.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny)

            y = 1 / (1 + np.exp(-feat))  # sigmoid
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                           self.grid[i]) * int(self.stride[i])
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, -1, nc))
        z = np.concatenate(z, axis=1)
        return z

    def __call__(self, preds_batch, org_size_batch, ratios_batch, txy_batch):
        """
        post-processing
        :param preds_batch:     list of predictions in a batch
        :param org_size_batch:  list of (org_img_w, org_img_h) in a batch
        :param ratios_batch:    list of (ratio_x, ratio_y) in a batch when resize-and-center-padding
        :param txy_batch:       list of (tx, ty) in a batch when resize-and-center-padding
        :return:
        """
        # print(f"Debug: preds_batch类型: {type(preds_batch)}, 长度: {len(preds_batch) if isinstance(preds_batch, list) else 'N/A'}")
        
        if isinstance(preds_batch, list) and len(preds_batch) == 3:
            # 3 output
            print("Debug: 使用3输出解码")
            dets = self.decode_for_3outputs(preds_batch)
        elif isinstance(preds_batch, list) and len(preds_batch) == 1:
            # 1 output - 针对PCB检测的特殊处理
            print("Debug: 使用1输出解码")
            pred = preds_batch[0]
            print(f"Debug: 原始输出shape: {pred.shape}")
            
            # 检查输出格式 (1, 8, 1278, 718)
            if len(pred.shape) == 4 and pred.shape[1] == 8:
                print("Debug: 检测到特征图格式，使用YOLOv5标准解码")
                
                # 将特征图reshape为标准的YOLOv5格式
                # (1, 8, 1278, 718) -> (1, 1278, 718, 8) -> (1, 1278*718, 8)
                pred = pred.transpose(0, 2, 3, 1)  # (1, 1278, 718, 8)
                pred = pred.reshape(1, -1, 8)  # (1, 917604, 8)
                
                print(f"Debug: 重塑后shape: {pred.shape}")
                
                # 应用sigmoid激活
                pred_sigmoid = 1 / (1 + np.exp(-pred))
                
                # 创建网格坐标
                h, w = 1278, 718
                grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)  # (H*W, 2)
                
                # 解码坐标 - 使用YOLOv5的标准解码方式
                # 假设stride=1 (因为特征图尺寸接近输入尺寸)
                stride = 1.0
                
                # 解码中心点坐标
                xy = (pred_sigmoid[0, :, 0:2] * 2.0 - 0.5 + grid) * stride
                
                # 解码宽高 - 使用更小的anchor
                anchor_wh = np.array([16, 16])  # 小anchor适合PCB缺陷检测
                wh = (pred_sigmoid[0, :, 2:4] * 2) ** 2 * anchor_wh
                
                # 置信度和类别
                conf = pred_sigmoid[0, :, 4]
                class_probs = pred_sigmoid[0, :, 5:]
                
                # 过滤低置信度
                conf_mask = conf > self.conf_thresh
                
                print(f"Debug: 置信度过滤前: {len(conf)} 个候选")
                print(f"Debug: 置信度过滤后: {np.sum(conf_mask)} 个候选")
                
                if np.sum(conf_mask) == 0:
                    results = [np.zeros((0, 6))] * len(org_size_batch)
                else:
                    # 应用过滤
                    xy_filtered = xy[conf_mask]
                    wh_filtered = wh[conf_mask]
                    conf_filtered = conf[conf_mask]
                    class_probs_filtered = class_probs[conf_mask]
                    
                    # 限制检测数量
                    max_detections = min(100, len(xy_filtered))
                    if len(xy_filtered) > max_detections:
                        # 按置信度排序
                        sorted_indices = np.argsort(-conf_filtered)[:max_detections]
                        xy_filtered = xy_filtered[sorted_indices]
                        wh_filtered = wh_filtered[sorted_indices]
                        conf_filtered = conf_filtered[sorted_indices]
                        class_probs_filtered = class_probs_filtered[sorted_indices]
                    
                    print(f"Debug: 最终处理 {len(xy_filtered)} 个检测")
                    
                    # 计算类别
                    if class_probs_filtered.shape[1] >= 2:
                        class_ids = np.argmax(class_probs_filtered[:, :2], axis=1)
                        class_scores = np.max(class_probs_filtered[:, :2], axis=1)
                    else:
                        class_ids = np.zeros(len(xy_filtered), dtype=int)
                        class_scores = np.ones(len(xy_filtered))
                    
                    # 计算最终置信度
                    final_conf = conf_filtered * class_scores
                    
                    # 再次过滤
                    final_mask = final_conf > self.conf_thresh
                    if not np.any(final_mask):
                        results = [np.zeros((0, 6))] * len(org_size_batch)
                    else:
                        xy_final = xy_filtered[final_mask]
                        wh_final = wh_filtered[final_mask]
                        final_conf_final = final_conf[final_mask]
                        class_ids_final = class_ids[final_mask]
                        
                        # 转换为xyxy格式
                        x1 = xy_final[:, 0] - wh_final[:, 0] / 2
                        y1 = xy_final[:, 1] - wh_final[:, 1] / 2
                        x2 = xy_final[:, 0] + wh_final[:, 0] / 2
                        y2 = xy_final[:, 1] + wh_final[:, 1] / 2
                        
                        # 限制在有效范围内
                        x1 = np.clip(x1, 0, self.net_w)
                        y1 = np.clip(y1, 0, self.net_h)
                        x2 = np.clip(x2, 0, self.net_w)
                        y2 = np.clip(y2, 0, self.net_h)
                        
                        # 过滤太小的框
                        valid_mask = (x2 - x1 > 5) & (y2 - y1 > 5)
                        if not np.any(valid_mask):
                            results = [np.zeros((0, 6))] * len(org_size_batch)
                        else:
                            detections = np.column_stack([
                                x1[valid_mask],
                                y1[valid_mask], 
                                x2[valid_mask],
                                y2[valid_mask],
                                final_conf_final[valid_mask],
                                class_ids_final[valid_mask]
                            ])
                            
                            print(f"Debug: 最终得到 {len(detections)} 个有效检测")
                            results = [detections] * len(org_size_batch)
            else:
                # 标准检测框格式
                print("Debug: 检测到标准检测框格式")
                dets = np.concatenate(preds_batch)
                # 使用原始NMS
                results = self.nms.non_max_suppression(
                    dets,
                    conf_thres=self.conf_thresh,
                    iou_thres=self.nms_thresh,
                    classes=None,
                    agnostic=self.agnostic_nms,
                    multi_label=self.multi_label,
                    max_det=self.max_det,
                )
        else:
            print('preds_batch type: {}'.format(type(preds_batch)))
            raise NotImplementedError

        # print(f"Debug: 最终结果数量: {[len(r) for r in results]}")

        # Rescale boxes from img_size to im0 size
        for det, (org_w, org_h), ratio, (tx1, ty1) in zip(results, org_size_batch, ratios_batch, txy_batch):
            if len(det):
                # print(f"Debug: 处理检测框 - 原图尺寸: {org_w}x{org_h}, 比例: {ratio}, 偏移: ({tx1}, {ty1})")
                # print(f"Debug: 转换前坐标: {det[:3, :4] if len(det) >= 3 else det[:, :4]}")
                
                # Rescale boxes from img_size to im0 size
                coords = det[:, :4].copy()  # 创建副本避免内存问题
                coords[:, [0, 2]] -= tx1  # x padding
                coords[:, [1, 3]] -= ty1  # y padding
                coords[:, [0, 2]] /= ratio[0]
                coords[:, [1, 3]] /= ratio[1]

                coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, org_w - 1)  # x1, x2
                coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, org_h - 1)  # y1, y2

                det[:, :4] = coords.round()

                # print(f"Debug: 转换后坐标: {det[:3, :4] if len(det) >= 3 else det[:, :4]}")

        return results


# numpy multiclass nms implementation from original yolov5 repo torch implementation
class pseudo_torch_nms:
    def nms_boxes(self, boxes, scores, iou_thres):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.copy() if isinstance(x, np.ndarray) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def box_area(self, box):
        # box = xyxy(4,n)
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
        # inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        inter = (np.min([a2, b2], 0) - np.max([a1, b1], 0)).clip(min=0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / (self.box_area(box1.T)[:, None] + self.box_area(box2.T) - inter)

    def nms(self, pred, conf_thres=0.25, iou_thres=0.5, agnostic=False, max_det=1000):
        return self.non_max_suppression(pred, conf_thres, iou_thres,
                                        classes=None,
                                        agnostic=agnostic,
                                        multi_label=True,
                                        max_det=max_det)
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                            agnostic=False, multi_label=False, max_det=300):
        """
        非极大值抑制
        Args:
            prediction: (batch_size, num_boxes, 7) 其中7为: [batch_id, x, y, w, h, conf, class_id]
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            classes: 类别过滤
            agnostic: 是否进行类别无关的NMS
            multi_label: 是否允许多标签
            max_det: 最大检测框数量
        Returns:
            list of detections, on (n,6) tensor per image [x, y, w, h, conf, class_id]
        """
        
        if not isinstance(prediction, np.ndarray):
            prediction = np.array(prediction)
            
        if len(prediction.shape) != 3:
            print(f"Warning: prediction shape应该是3维的 [batch_size, num_boxes, 7]，当前shape: {prediction.shape}")
            # 如果是2维的，添加batch维度
            if len(prediction.shape) == 2:
                prediction = prediction[None]

        bs = prediction.shape[0]  # batch size
        if bs == 0:
            return [np.zeros((0, 6))]

        # Settings
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

        output = [np.zeros((0, 6))] * bs
        
        for xi, x in enumerate(prediction):  # image index, image inference
            # 如果没有框，跳过
            if len(x) == 0:
                continue

            # 计算置信度
            x = x[x[:, 5] > conf_thres]  # 基于置信度过滤
            
            # 如果没有框，跳过
            if len(x) == 0:
                continue
                
            # 计算框的面积
            box = x[:, 1:5].copy()  # boxes [x, y, w, h]
            area = box[:, 2] * box[:, 3]  # area = w * h
            
            # 按置信度排序
            i = np.argsort(-x[:, 5])
            x = x[i]
            area = area[i]

            # 开始NMS
            selected_indices = []
            while len(x) > 0:
                # 保留分数最高的框
                selected_indices.append(i[0])
                if len(x) == 1:
                    break
                    
                # 计算IoU
                box1 = box[i[0]]
                box2 = box[i[1:]]
                ious = self.box_iou_numpy(box1, box2)
                
                # 移除重叠的框
                overlapped = ious > iou_thres
                i = i[1:][~overlapped]
                x = x[~overlapped]
                area = area[~overlapped]
            
            if len(selected_indices) > max_det:
                selected_indices = selected_indices[:max_det]
                
            # 保存结果
            output[xi] = prediction[xi][selected_indices][:, 1:]  # 移除batch_id

        return output
        
    def box_iou_numpy(self, box1, boxes):
        """计算一个框和多个框之间的IoU
        Args:
            box1: 单个框 [x, y, w, h]
            boxes: 多个框 [N, x, y, w, h]
        Returns:
            IoU values
        """
        # 转换为左上右下格式
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        
        b2_x1, b2_y1 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
        b2_x2, b2_y2 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
        
        # 计算交集
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # 计算并集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-16)

    def non_max_suppression_custom(self, boxes, conf_thres=0.25, iou_thres=0.45):
        print('DEBUG: non_max_suppression_custom DEFINED')
        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
            
        # 按置信度排序
        boxes = boxes[np.argsort(-boxes[:, 4])]
        
        keep_boxes = []
        while len(boxes) > 0:
            # 取分数最高的框作为基准
            curr_box = boxes[0]
            keep_boxes.append(curr_box)
            if len(boxes) == 1:
                break
            # 计算其余框与当前框的IoU
            ious = self.box_iou_numpy(curr_box[:4], boxes[1:, :4])
            # 保留 IoU 小于阈值的框
            boxes = boxes[1:][ious <= iou_thres]
        
        # 返回保留的框
        print('DEBUG: non_max_suppression_custom CALLED')
        return np.stack(keep_boxes) if len(keep_boxes) else np.zeros((0, 6), dtype=np.float32)
        
    def box_iou_numpy(self, box1, boxes):
        """计算一个框和多个框之间的IoU
        Args:
            box1: 单个框 [x, y, w, h]
            boxes: 多个框 [N, x, y, w, h]
        Returns:
            IoU values [N]
        """
        # 转换为左上右下格式
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        
        b2_x1, b2_y1 = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
        b2_x2, b2_y2 = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
        
        # 计算交集
        inter_x1 = np.maximum(b1_x1, b2_x1)
        inter_y1 = np.maximum(b1_y1, b2_y1)
        inter_x2 = np.minimum(b1_x2, b2_x2)
        inter_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # 计算并集
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-16)
