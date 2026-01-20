#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTSP输出检测器 - 优化版本
包含跳帧处理、延迟优化和性能提升功能
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import threading
from datetime import datetime

# 设置环境
sophon_opencv_path = "/opt/sophon/sophon-opencv_1.9.0/opencv-python"
if sophon_opencv_path not in sys.path:
    sys.path.insert(0, sophon_opencv_path)

import sophon.sail as sail

try:
    from python.postprocess_yolov8 import PostProcess, non_max_suppression, nms_numpy
except ImportError:
    from postprocess_yolov8 import PostProcess, non_max_suppression, nms_numpy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedRTSPDetector:
    """优化的RTSP检测器"""
    
    def __init__(self, config):
        # 基本配置
        self.model_path = config.model_path
        self.input_rtsp = config.input_rtsp
        self.output_rtsp = config.output_rtsp
        self.device_id = config.device_id
        self.conf_thresh = config.conf_thresh
        self.nms_thresh = config.nms_thresh
        
        # 优化参数
        self.frame_skip = getattr(config, 'frame_skip', 2)  # 跳帧数：处理1帧，跳过2帧
        self.max_queue_size = getattr(config, 'max_queue_size', 3)  # 最大队列大小
        self.detection_interval = getattr(config, 'detection_interval', 1)  # 检测间隔
        self.low_latency_mode = getattr(config, 'low_latency_mode', True)  # 低延迟模式
        
        # 输出设置
        self.output_width = getattr(config, 'output_width', 1920)
        self.output_height = getattr(config, 'output_height', 1080)
        self.output_fps = getattr(config, 'output_fps', 15)  # 降低输出帧率
        
        # 运行状态
        self.running = False
        self.frame_count = 0
        self.processed_count = 0
        self.last_detection_result = None
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'detection_time': 0,
            'encode_time': 0,
            'total_time': 0,
            'current_fps': 0,
            'detection_count': 0,
            'last_update': time.time()
        }
        
        # 统计文件路径
        self.stats_file = 'logs/rtsp_stats.json'
        
        # 初始化组件
        self.init_sophon_sdk()
        self.init_model()
        
    def init_sophon_sdk(self):
        """初始化Sophon SDK"""
        try:
            self.handle = sail.Handle(self.device_id)
            self.bmcv = sail.Bmcv(self.handle)
            logging.info(f"Sophon SDK初始化成功，设备ID: {self.device_id}")
        except Exception as e:
            logging.error(f"Sophon SDK初始化失败: {e}")
            raise
    
    def init_model(self):
        """初始化模型"""
        try:
            # 加载模型
            self.engine = sail.Engine(self.model_path, self.device_id, sail.IOMode.SYSO)
            self.graph_name = self.engine.get_graph_names()[0]
            
            # 获取输入输出信息
            self.input_name = self.engine.get_input_names(self.graph_name)[0]
            self.output_names = self.engine.get_output_names(self.graph_name)
            
            # 获取输入形状和数据类型
            self.input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
            self.input_dtype = self.engine.get_input_dtype(self.graph_name, self.input_name)
            self.input_scale = self.engine.get_input_scale(self.graph_name, self.input_name)
            
            # 获取BMImage数据类型
            self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
            
            # 修正网络尺寸
            if len(self.input_shape) == 4:
                self.net_h = self.input_shape[2]
                self.net_w = self.input_shape[3]
            else:
                self.net_h = self.net_w = 640
            
            # 计算缩放系数
            self.ab = [x * self.input_scale / 255.0 for x in [1, 0, 1, 0, 1, 0]]
            
            logging.info(f"网络尺寸: {self.net_w}x{self.net_h}")
            logging.info(f"输入张量形状: {self.input_shape}, 数据类型: {self.input_dtype}")
            logging.info(f"输入缩放: {self.input_scale}, 缩放系数: {self.ab}")
            
            # 创建输入张量（预分配）
            self.input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, True, True)
            
            # 创建输出张量
            self.output_tensors = {}
            for output_name in self.output_names:
                output_shape = self.engine.get_output_shape(self.graph_name, output_name)
                output_dtype = self.engine.get_output_dtype(self.graph_name, output_name)
                output_tensor = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
                self.output_tensors[output_name] = output_tensor
                logging.info(f"输出 {output_name}: {output_shape}")
            
            # 初始化后处理 - 适配YOLOv8
            self.postprocess = PostProcess(
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
                agnostic=False
            )
            
            logging.info(f"模型加载成功: {self.model_path}")
            
        except Exception as e:
            logging.error(f"模型初始化失败: {e}")
            raise
    
    def create_multi_decoder(self):
        """创建MultiDecoder - 优化版本"""
        try:
            logging.info(f"连接输入RTSP流: {self.input_rtsp}")
            
            # 低延迟配置 - 优化参数以减少断连
            if self.low_latency_mode:
                queue_size = 2  # 增加到2，提供缓冲避免频繁断连
                discard_mode = 1  # 启用丢帧模式
            else:
                queue_size = self.max_queue_size
                discard_mode = 0
            
            # 创建MultiDecoder
            self.multi_decoder = sail.MultiDecoder(
                queue_size=queue_size,
                tpu_id=self.device_id,
                discard_mode=discard_mode
            )
            
            # 设置为网络视频流模式
            self.multi_decoder.set_local_flag(False)
            self.multi_decoder.set_read_timeout(3)  # 增加超时时间到3秒，避免网络波动导致断连
            
            # 添加视频通道
            self.channel_idx = self.multi_decoder.add_channel(self.input_rtsp)
            if self.channel_idx < 0:
                logging.error(f"无法添加输入视频通道: {self.input_rtsp}")
                return False
            
            logging.info(f"输入视频通道添加成功，通道索引: {self.channel_idx}")
            
            # 等待连接稳定
            time.sleep(1)
            
            # 获取输入视频信息
            try:
                frame_shape = self.multi_decoder.get_frame_shape(self.channel_idx)
                fps = self.multi_decoder.get_channel_fps(self.channel_idx)
                logging.info(f"输入视频帧尺寸: {frame_shape}")
                logging.info(f"输入视频帧率: {fps}")
                
                # 动态调整输出分辨率
                if frame_shape and len(frame_shape) == 4:
                    h_in = frame_shape[2]
                    w_in = frame_shape[3]
                    if (self.output_width != w_in) or (self.output_height != h_in):
                        logging.info(f"自动调整输出分辨率为 {w_in}x{h_in}")
                        self.output_width = w_in
                        self.output_height = h_in
                        
            except Exception as e:
                logging.warning(f"获取输入视频信息失败: {e}")
            
            return True
            
        except Exception as e:
            logging.error(f"创建MultiDecoder失败: {e}")
            return False
    
    def preprocess_bmcv(self, input_bmimg):
        """完整的图像预处理（参考原始代码）"""
        # 1. 转换为RGB_PLANAR格式
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                      sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        
        # 2. 缩放和padding
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        
        # 3. 转换数据类型和缩放
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, 
                                         sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, 
                           ((self.ab[0], self.ab[1]), (self.ab[2], self.ab[3]), (self.ab[4], self.ab[5])))
        
        return preprocessed_bmimg, ratio, txy

    def resize_bmcv(self, bmimg):
        """图像缩放（参考原始代码）"""
        img_w = bmimg.width()
        img_h = bmimg.height()
        
        # 计算缩放比例
        r_w = self.net_w / img_w
        r_h = self.net_h / img_h
        
        if r_h > r_w:
            tw = self.net_w
            th = int(r_w * img_h)
            tx1 = tx2 = 0
            ty1 = int((self.net_h - th) / 2)
            ty2 = self.net_h - th - ty1
        else:
            tw = int(r_h * img_w)
            th = self.net_h
            tx1 = int((self.net_w - tw) / 2)
            tx2 = self.net_w - tw - tx1
            ty1 = ty2 = 0

        ratio = (min(r_w, r_h), min(r_w, r_h))
        txy = (tx1, ty1)
        
        # 设置padding属性
        attr = sail.PaddingAtrr()
        attr.set_stx(tx1)
        attr.set_sty(ty1)
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(114)
        attr.set_g(114)
        attr.set_b(114)
        
        # 缩放和padding
        resized_img_rgb = self.bmcv.crop_and_resize_padding(
            bmimg, 0, 0, img_w, img_h, self.net_w, self.net_h, attr
        )
        
        return resized_img_rgb, ratio, txy
    
    def process_frame(self, bmimg):
        """处理单帧"""
        try:
            # 完整的预处理
            preprocessed_img, ratio, offset = self.preprocess_bmcv(bmimg)
            
            # 转换为tensor（使用预分配的张量）
            self.bmcv.bm_image_to_tensor(preprocessed_img, self.input_tensor)
            
            # 推理
            input_tensors = {self.input_name: self.input_tensor}
            self.engine.process(self.graph_name, input_tensors, {self.input_name: self.input_shape}, self.output_tensors)
            
            # 获取输出
            outputs = []
            for output_name in self.output_names:
                output_data = self.output_tensors[output_name].asnumpy()
                outputs.append(output_data)
            
            # YOLOv8后处理 - 转换输出格式
            img_shape = (bmimg.width(), bmimg.height())
            results = self.postprocess_yolov8(outputs, img_shape, ratio, offset)
            
            return results
            
        except Exception as e:
            logging.error(f"处理帧失败: {e}")
            return None
    
    def postprocess_yolov8(self, outputs, img_shape, ratio, offset):
        """YOLOv8专用后处理 - 针对2类PCB检测优化"""
        try:
            if not outputs:
                return None
            
            # 获取第一个输出
            prediction = outputs[0]  # shape: (1, 6, 8400)
            
            # 打印调试信息
            if self.processed_count <= 3:
                logging.info(f"YOLOv8输出形状: {prediction.shape}")
                logging.info(f"原始输出的前几个值: {prediction.flatten()[:10]}")
            
            # YOLOv8输出格式: [1, 6, 8400] -> [1, 8400, 6]
            # 转置: [batch, features, anchors] -> [batch, anchors, features]
            prediction = np.transpose(prediction, (0, 2, 1))  # -> [1, 8400, 6]
            
            if self.processed_count <= 3:
                logging.info(f"转置后形状: {prediction.shape}")
            
            # 处理单个批次
            pred = prediction[0]  # shape: [8400, 6]
            
            # YOLOv8格式: [x, y, w, h, class1_prob, class2_prob]
            boxes = pred[:, :4]  # [8400, 4] - 坐标
            class_probs = pred[:, 4:]  # [8400, 2] - 类别概率
            
            # 计算最大类别概率和对应的类别ID
            max_class_probs = np.max(class_probs, axis=1)  # [8400]
            class_ids = np.argmax(class_probs, axis=1)  # [8400]
            
            # 置信度过滤
            conf_mask = max_class_probs > self.conf_thresh
            if not np.any(conf_mask):
                return None
            
            # 筛选有效检测
            valid_boxes = boxes[conf_mask]  # [N, 4]
            valid_scores = max_class_probs[conf_mask]  # [N]
            valid_class_ids = class_ids[conf_mask]  # [N]
            
            if len(valid_boxes) == 0:
                return None
            
            # 转换坐标格式：中心点格式 -> 左上右下格式
            x_center, y_center, width, height = valid_boxes.T
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            xyxy_boxes = np.column_stack([x1, y1, x2, y2])
            
            # 应用NMS
            keep_indices = nms_numpy(xyxy_boxes, valid_scores, self.nms_thresh)
            
            if len(keep_indices) == 0:
                return None
            
            # 获取最终检测结果
            final_boxes = xyxy_boxes[keep_indices]
            final_scores = valid_scores[keep_indices]
            final_class_ids = valid_class_ids[keep_indices]
            
            # 坐标转换：从网络尺寸转换到原始图像尺寸
            img_w, img_h = img_shape
            scale_x, scale_y = ratio
            offset_x, offset_y = offset
            
            # 移除padding偏移
            final_boxes[:, 0] -= offset_x  # x1
            final_boxes[:, 1] -= offset_y  # y1  
            final_boxes[:, 2] -= offset_x  # x2
            final_boxes[:, 3] -= offset_y  # y2
            
            # 缩放到原始图像尺寸
            final_boxes[:, 0] /= scale_x
            final_boxes[:, 1] /= scale_y
            final_boxes[:, 2] /= scale_x
            final_boxes[:, 3] /= scale_y
            
            # 限制坐标范围
            final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, img_w)
            final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, img_h) 
            final_boxes[:, 2] = np.clip(final_boxes[:, 2], 0, img_w)
            final_boxes[:, 3] = np.clip(final_boxes[:, 3], 0, img_h)
            
            # 组合最终结果: [x1, y1, x2, y2, score, class_id]
            detections = np.column_stack([
                final_boxes,
                final_scores,
                final_class_ids.astype(np.float32)
            ])
            
            if self.processed_count <= 3:
                logging.info(f"检测到 {len(detections)} 个目标")
                if len(detections) > 0:
                    logging.info(f"第一个检测结果: {detections[0]}")
            
            return detections
                
        except Exception as e:
            logging.error(f"YOLOv8后处理失败: {e}")
            import traceback
            logging.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def draw_detections_fast(self, bmimg, detections):
        """快速绘制检测结果（包含类别标签）"""
        if detections is None or len(detections) == 0:
            return bmimg
        
        try:
            # 转换为BGR格式用于绘制
            bgr_img = self.bmcv.convert_format(bmimg)
            
            # 类别名称
            class_names = ["link", "unknown"]  # 0: link(连锡), 1: unknown(未知)
            
            # 绘制检测框和标签
            for det in detections:
                x1, y1, x2, y2, score, class_id = det[:6]
                
                # 确保坐标在有效范围内
                x1 = max(0, min(int(x1), bmimg.width() - 1))
                y1 = max(0, min(int(y1), bmimg.height() - 1))
                x2 = max(0, min(int(x2), bmimg.width() - 1))
                y2 = max(0, min(int(y2), bmimg.height() - 1))
                
                # 跳过无效框
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # 获取类别信息
                class_idx = int(class_id)
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                confidence = float(score)
                
                # 设置颜色：link(连锡)用红色，unknown(未知)用绿色
                if class_idx == 0:  # link
                    color = (0, 0, 255)  # 红色
                else:  # unknown
                    color = (0, 255, 0)  # 绿色
                
                thickness = 2
                
                try:
                    # 只绘制检测框，避免任何额外绘制导致BMCV错误
                    self.bmcv.rectangle(bgr_img, x1, y1, x2-x1, y2-y1, color, thickness)
                        
                except Exception as e:
                    logging.debug(f"绘制检测框失败: {e}")
                    continue
            
            return bgr_img
            
        except Exception as e:
            logging.error(f"绘制检测结果失败: {e}")
            return bmimg
    
    def detection_loop(self):
        """主检测循环 - 优化版本"""
        logging.info("开始检测循环...")
        
        # 创建编码器
        enc_fmt = "h264_bm"
        pix_fmt = "NV12"
        enc_params = f"width={self.output_width}:height={self.output_height}:bitrate=1500:gop=16:gop_preset=2:framerate={self.output_fps}"
        encoder = sail.Encoder(self.output_rtsp, self.handle, enc_fmt, pix_fmt, enc_params, 1, 0)
        
        # 预分配缓冲区
        nv12_buffer = sail.BMImage(self.handle, self.output_height, self.output_width, sail.Format.FORMAT_NV12, sail.DATA_TYPE_EXT_1N_BYTE)
        
        # 性能统计
        last_stats_time = time.time()
        detection_times = []
        encode_times = []
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. 读取帧 - 增加超时时间和错误处理
                bmimg = sail.BMImage()
                ret = self.multi_decoder.read(self.channel_idx, bmimg, 500)  # 增加超时到500ms
                
                if ret != 0:
                    if ret == -1:
                        # 超时，可能是网络波动，继续重试
                        continue
                    else:
                        logging.warning(f"读取帧失败 (错误码: {ret})")
                        time.sleep(0.05)  # 等待稍长时间再重试
                        continue
                
                if bmimg.width() == 0 or bmimg.height() == 0:
                    continue
                
                self.frame_count += 1
                self.stats['total_frames'] += 1
                
                # 2. 跳帧处理
                should_process = (self.frame_count % (self.frame_skip + 1) == 0)
                
                if should_process:
                    # 执行检测
                    detection_start = time.time()
                    detections = self.process_frame(bmimg)
                    detection_time = time.time() - detection_start
                    
                    detection_times.append(detection_time * 1000)
                    if len(detection_times) > 100:
                        detection_times.pop(0)
                    
                    # 更新最后的检测结果
                    self.last_detection_result = detections
                    self.processed_count += 1
                    self.stats['processed_frames'] += 1
                    self.stats['detection_time'] += detection_time
                    
                    # 调试输出
                    if self.processed_count % 10 == 0:
                        if detections is not None and len(detections) > 0:
                            logging.info(f"帧 {self.frame_count}: 检测到 {len(detections)} 个目标")
                        else:
                            logging.info(f"帧 {self.frame_count}: 无检测目标")
                else:
                    # 跳过检测，使用上一次的结果
                    detections = self.last_detection_result
                    self.stats['skipped_frames'] += 1
                
                # 3. 调整图像尺寸
                if bmimg.width() != self.output_width or bmimg.height() != self.output_height:
                    resized_img = self.bmcv.resize(bmimg, self.output_width, self.output_height)
                else:
                    resized_img = bmimg
                
                # 4. 绘制检测结果
                if detections is not None and len(detections) > 0:
                    output_img = self.draw_detections_fast(resized_img, detections)
                    # 更新检测目标数
                    self.stats['detection_count'] = len(detections)
                else:
                    output_img = resized_img
                    self.stats['detection_count'] = 0
                
                # 5. 编码和推流
                encode_start = time.time()
                
                # 转换为NV12格式
                if output_img.format() != sail.Format.FORMAT_NV12:
                    self.bmcv.convert_format(output_img, nv12_buffer)
                else:
                    nv12_buffer = output_img
                
                # 推流
                encoder.video_write(nv12_buffer)
                
                encode_time = time.time() - encode_start
                encode_times.append(encode_time * 1000)
                if len(encode_times) > 100:
                    encode_times.pop(0)
                
                self.stats['encode_time'] += encode_time
                
                # 6. 性能统计
                total_time = time.time() - loop_start
                self.stats['total_time'] += total_time
                
                current_time = time.time()
                if current_time - last_stats_time >= 2.0:  # 每2秒输出一次统计
                    if detection_times and encode_times:
                        avg_detection = sum(detection_times) / len(detection_times)
                        avg_encode = sum(encode_times) / len(encode_times)
                        
                        # 计算实际FPS
                        time_window = current_time - last_stats_time
                        actual_fps = self.stats['total_frames'] / time_window if time_window > 0 else 0
                        
                        logging.info(f"FPS: {actual_fps:.1f} | 检测: {avg_detection:.1f}ms | 编码: {avg_encode:.1f}ms | 处理率: {self.stats['processed_frames']}/{self.stats['total_frames']}")
                        
                        # 更新统计信息并保存到文件
                        self.stats['current_fps'] = round(actual_fps, 1)
                        self.stats['last_update'] = current_time
                        self.save_stats()
                        
                        # 重置统计
                        detection_count_backup = self.stats.get('detection_count', 0)
                        self.stats = {k: 0 for k in self.stats.keys()}
                        self.stats['detection_count'] = detection_count_backup
                        self.stats['current_fps'] = round(actual_fps, 1)
                        self.stats['last_update'] = current_time
                        last_stats_time = current_time
                
                # 7. 帧率控制
                if self.output_fps > 0:
                    target_frame_time = 1.0 / self.output_fps
                    elapsed = time.time() - loop_start
                    if elapsed < target_frame_time:
                        time.sleep(target_frame_time - elapsed)
                
            except Exception as e:
                logging.error(f"检测循环错误: {e}")
                time.sleep(0.1)
                continue
        
        # 清理资源
        try:
            encoder.release()
        except:
            pass
        
        logging.info("检测循环结束")
    
    def start(self):
        """启动检测器"""
        try:
            # 创建解码器
            if not self.create_multi_decoder():
                return False
            
            # 启动检测循环
            self.running = True
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            logging.info("=== RTSP输出检测器已启动 ===")
            logging.info(f"输入流: {self.input_rtsp}")
            logging.info(f"输出流: {self.output_rtsp}")
            logging.info(f"跳帧设置: 处理1帧，跳过{self.frame_skip}帧")
            logging.info(f"低延迟模式: {self.low_latency_mode}")
            logging.info("在VLC中打开输出流地址即可观看检测结果")
            logging.info("按 Ctrl+C 停止...")
            logging.info("=" * 40)
            
            return True
            
        except Exception as e:
            logging.error(f"启动检测器失败: {e}")
            return False
    
    def save_stats(self):
        """保存统计信息到文件"""
        try:
            import json
            stats_data = {
                'fps': self.stats.get('current_fps', 0),
                'detection_count': self.stats.get('detection_count', 0),
                'processed_frames': self.stats.get('processed_frames', 0),
                'total_frames': self.stats.get('total_frames', 0),
                'last_update': self.stats.get('last_update', time.time())
            }
            
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            
            with open(self.stats_file, 'w') as f:
                json.dump(stats_data, f)
        except Exception as e:
            logging.error(f"保存统计信息失败: {e}")
    
    def stop(self):
        """停止检测器"""
        logging.info("正在停止检测器...")
        self.running = False
        
        # 等待线程结束
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=5)
        
        # 清理统计文件
        try:
            import os
            if os.path.exists(self.stats_file):
                os.remove(self.stats_file)
        except:
            pass
        
        # 清理资源
        try:
            if hasattr(self, 'multi_decoder'):
                del self.multi_decoder
        except:
            pass
        
        logging.info("检测器已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化的RTSP检测器')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input_rtsp', type=str, required=True, help='输入RTSP流地址')
    parser.add_argument('--output_rtsp', type=str, required=True, help='输出RTSP流地址')
    parser.add_argument('--device_id', type=int, default=0, help='设备ID')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='NMS阈值')
    
    # 优化参数
    parser.add_argument('--frame_skip', type=int, default=2, help='跳帧数量(处理1帧跳过N帧)')
    parser.add_argument('--max_queue_size', type=int, default=3, help='最大队列大小')
    parser.add_argument('--output_fps', type=int, default=15, help='输出帧率')
    parser.add_argument('--low_latency_mode', action='store_true', help='启用低延迟模式')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = OptimizedRTSPDetector(args)
    
    try:
        # 启动检测器
        if detector.start():
            # 等待用户中断
            while True:
                time.sleep(1)
        else:
            logging.error("启动检测器失败")
            return 1
            
    except KeyboardInterrupt:
        logging.info("收到停止信号...")
    except Exception as e:
        logging.error(f"运行错误: {e}")
        return 1
    finally:
        detector.stop()
        logging.info("程序结束")
    
    return 0

if __name__ == "__main__":
    exit(main()) 