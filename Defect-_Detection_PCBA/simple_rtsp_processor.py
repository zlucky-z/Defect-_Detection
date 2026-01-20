#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版RTSP流PCB缺陷检测处理器
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import threading
import tempfile
import numpy as np
import sophon.sail as sail
from python.postprocess_numpy import PostProcess
from python.utils import COLORS
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleRTSPProcessor:
    def __init__(self, args):
        self.args = args
        self.input_rtsp = args.input_rtsp
        self.output_rtsp = args.output_rtsp
        self.bmodel_path = args.bmodel
        self.dev_id = args.dev_id
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        
        # 初始化统计
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # 初始化设备
        self.handle = sail.Handle(self.dev_id)
        self.bmcv = sail.Bmcv(self.handle)
        
        # 初始化模型
        self.init_yolov5()
        
        # 初始化推流
        self.frame_queue = queue.Queue(maxsize=10)
        self.ffmpeg_process = None
        self.streaming_thread = None

    def init_yolov5(self):
        """初始化YOLOv5模型"""
        self.net = sail.Engine(self.bmodel_path, self.dev_id, sail.IOMode.SYSIO)
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        
        # 获取输入输出信息
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.input_dtype = self.net.get_input_dtype(self.graph_name, self.input_name)
        self.img_dtype = self.bmcv.get_bm_image_data_format(self.input_dtype)
        self.input_scale = self.net.get_input_scale(self.graph_name, self.input_name)
        
        # 设置预处理参数
        self.ab = [x * self.input_scale / 255. for x in [1, 0, 1, 0, 1, 0]]
        
        self.output_tensors = {}
        for output_name in self.output_names:
            output_shape = self.net.get_output_shape(self.graph_name, output_name)
            output_dtype = self.net.get_output_dtype(self.graph_name, output_name)
            self.output_tensors[output_name] = sail.Tensor(self.handle, output_shape, output_dtype, True, True)
        
        # 网络输入尺寸
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        # 初始化后处理
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=False,
            multi_label=True,
            max_det=1000,
            net_h=self.net_h,
            net_w=self.net_w
        )
        
        logging.info(f"YOLOv5模型初始化完成: {self.input_shape}")

    def preprocess_bmcv(self, input_bmimg):
        """预处理图像"""
        # 转换为RGB平面格式
        rgb_planar_img = sail.BMImage(self.handle, input_bmimg.height(), input_bmimg.width(),
                                     sail.Format.FORMAT_RGB_PLANAR, sail.DATA_TYPE_EXT_1N_BYTE)
        self.bmcv.convert_format(input_bmimg, rgb_planar_img)
        
        # 调整大小并填充
        resized_img_rgb, ratio, txy = self.resize_bmcv(rgb_planar_img)
        
        # 转换数据类型和归一化
        preprocessed_bmimg = sail.BMImage(self.handle, self.net_h, self.net_w, 
                                         sail.Format.FORMAT_RGB_PLANAR, self.img_dtype)
        self.bmcv.convert_to(resized_img_rgb, preprocessed_bmimg, 
                           ((self.ab[0], self.ab[1]), (self.ab[2], self.ab[3]), (self.ab[4], self.ab[5])))
        
        return preprocessed_bmimg, ratio, txy
    
    def resize_bmcv(self, bmimg):
        """调整图像大小"""
        img_w = bmimg.width()
        img_h = bmimg.height()
        
        # 计算缩放参数
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
        
        # 缩放和填充
        attr = sail.PaddingAtrr()
        attr.set_stx(tx1)
        attr.set_sty(ty1)
        attr.set_w(tw)
        attr.set_h(th)
        attr.set_r(114)
        attr.set_g(114)
        attr.set_b(114)
        
        resized_img_rgb = self.bmcv.crop_and_resize_padding(bmimg, 0, 0, img_w, img_h, 
                                                           self.net_w, self.net_h, attr)
        return resized_img_rgb, ratio, txy

    def detect_and_draw(self, bmimg):
        """检测并绘制结果"""
        # 预处理
        preprocessed_bmimg, ratio, txy = self.preprocess_bmcv(bmimg)
        
        # 准备输入tensor
        input_tensor = sail.Tensor(self.handle, self.input_shape, self.input_dtype, False, False)
        self.bmcv.bm_image_to_tensor(preprocessed_bmimg, input_tensor)
        
        # 推理
        input_tensors = {self.input_name: input_tensor}
        input_shapes = {self.input_name: self.input_shape}
        self.net.process(self.graph_name, input_tensors, input_shapes, self.output_tensors)
        
        # 获取输出
        outputs = []
        for name in self.output_names:
            output_data = self.output_tensors[name].asnumpy()
            outputs.append(output_data)
        
        # 后处理
        ori_size = (bmimg.width(), bmimg.height())
        results = self.postprocess(outputs, [ori_size], [ratio], [txy])
        detections = results[0] if results else np.zeros((0, 6))
        
        # 绘制检测结果
        img_bgr = self.bmcv.convert_format(bmimg)
        thickness = 2
        
        detection_count = 0
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            
            if conf < 0.25:
                continue
                
            detection_count += 1
            
            # 选择颜色
            color_index = (class_id % 2) + 1
            color = COLORS[color_index]
            
            # 绘制矩形框
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                self.bmcv.rectangle(img_bgr, x1, y1, x2 - x1, y2 - y1, color, thickness)
        
        return img_bgr, detection_count

    def start_ffmpeg_streaming(self):
        """启动FFmpeg实时推流 - 简化版本"""
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-r', '10',
            '-i', 'pipe:0',
            '-c:v', 'h264_bm',
            '-b:v', '1000k',
            '-r', '10',
            '-f', 'rtsp',
            self.output_rtsp
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logging.info(f"FFmpeg推流启动: {self.output_rtsp}")
            return self.ffmpeg_process
            
        except Exception as e:
            logging.error(f"启动FFmpeg推流失败: {e}")
            raise

    def streaming_worker(self):
        """推流工作线程"""
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=1)
                if frame_data is None:  # 退出信号
                    break
                    
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.write(frame_data)
                    self.ffmpeg_process.stdin.flush()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"推流错误: {e}")
                break

    def run(self):
        """运行主循环"""
        logging.info(f"开始处理RTSP流: {self.input_rtsp} -> {self.output_rtsp}")
        
        # 尝试多种方式打开输入流
        decoder = None
        retry_count = 0
        max_retries = 3
        
        while decoder is None and retry_count < max_retries:
            try:
                logging.info(f"尝试连接RTSP流 (第{retry_count + 1}次)...")
                decoder = sail.Decoder(self.input_rtsp, True, self.args.dev_id)
                
                if decoder.is_opened():
                    logging.info("RTSP流连接成功")
                    break
                else:
                    logging.warning(f"RTSP流连接失败，尝试次数: {retry_count + 1}")
                    decoder = None
                    
            except Exception as e:
                logging.error(f"连接RTSP流时出错: {e}")
                decoder = None
                
            retry_count += 1
            if retry_count < max_retries:
                logging.info("等待5秒后重试...")
                time.sleep(5)
        
        if decoder is None:
            logging.error(f"无法打开RTSP流: {self.input_rtsp}")
            logging.error("请检查:")
            logging.error("1. RTSP地址是否正确")
            logging.error("2. 网络连接是否正常")
            logging.error("3. 摄像头是否在线")
            return
        
        # 启动FFmpeg推流
        try:
            self.start_ffmpeg_streaming()
            
            # 启动推流工作线程
            self.streaming_thread = threading.Thread(target=self.streaming_worker)
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            
            logging.info("推流系统启动成功")
        except Exception as e:
            logging.error(f"启动推流系统失败: {e}")
            return
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        try:
            while consecutive_failures < max_consecutive_failures:
                frame = sail.BMImage()
                ret = decoder.read(self.handle, frame)
                
                if ret != 0:
                    consecutive_failures += 1
                    logging.warning(f"读取RTSP流失败 (连续失败: {consecutive_failures})")
                    time.sleep(0.5)
                    continue
                
                # 重置失败计数
                consecutive_failures = 0
                
                # 检测并绘制
                result_frame, det_count = self.detect_and_draw(frame)
                
                # 保存为JPEG格式用于推流
                temp_jpg_path = f"/tmp/rtsp_frame_{self.frame_count}.jpg"
                self.bmcv.imwrite(temp_jpg_path, result_frame)
                
                # 读取JPEG数据并添加到推流队列
                try:
                    with open(temp_jpg_path, 'rb') as f:
                        frame_bytes = f.read()
                    
                    self.frame_queue.put(frame_bytes, timeout=0.1)
                    
                    # 清理临时文件
                    os.remove(temp_jpg_path)
                    
                except queue.Full:
                    # 队列满了，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame_bytes, timeout=0.1)
                    except queue.Empty:
                        pass
                except Exception as e:
                    logging.error(f"帧处理错误: {e}")
                
                # 更新统计
                self.frame_count += 1
                if det_count > 0:
                    self.detection_count += 1
                    logging.info(f"帧 {self.frame_count}: 检测到 {det_count} 个目标")
                
                # 每100帧输出统计
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    logging.info(f"处理统计: 帧数={self.frame_count}, 检测帧数={self.detection_count}, FPS={fps:.2f}")
                
                # 控制帧率
                time.sleep(0.1)  # 约10fps
            
            if consecutive_failures >= max_consecutive_failures:
                logging.error("连续读取失败次数过多，停止处理")
                
        except KeyboardInterrupt:
            logging.info("收到停止信号")
        except Exception as e:
            logging.error(f"处理过程中出错: {e}")
        finally:
            # 停止推流
            if self.frame_queue:
                self.frame_queue.put(None)  # 发送退出信号
            if self.streaming_thread:
                self.streaming_thread.join(timeout=2)
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            if decoder:
                decoder.release()
            logging.info("清理完成")

def main():
    parser = argparse.ArgumentParser(description='简化版RTSP流PCB缺陷检测')
    parser.add_argument('--input_rtsp', type=str, required=True, help='输入RTSP流地址')
    parser.add_argument('--output_rtsp', type=str, required=True, help='输出RTSP流地址')
    parser.add_argument('--bmodel', type=str, default='models/yolov5s_16848_f16.bmodel', help='模型文件路径')
    parser.add_argument('--dev_id', type=int, default=0, help='设备ID')
    parser.add_argument('--conf_thresh', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='NMS阈值')
    
    args = parser.parse_args()
    
    logging.info("=== 简化版RTSP流PCB缺陷检测 ===")
    logging.info(f"输入流: {args.input_rtsp}")
    logging.info(f"输出流: {args.output_rtsp}")
    logging.info(f"模型: {args.bmodel}")
    logging.info(f"置信度阈值: {args.conf_thresh}")
    
    processor = SimpleRTSPProcessor(args)
    processor.run()

if __name__ == '__main__':
    main() 