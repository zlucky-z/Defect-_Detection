# 算能PCB缺陷检测RTSP推流系统

## 概述
本系统专为算能BM1688设备优化，基于算能官方FFmpeg文档进行配置，使用算能专用的硬件编码器进行RTSP视频流推送。

## 算能FFmpeg优化

### 1. 编码器选择
- **使用**: `h264_bm` (算能H.264硬件编码器)
- **优势**: 
  - 硬件加速编码，性能更好
  - 专为算能芯片优化
  - 功耗更低

### 2. 编码参数优化
根据算能FFmpeg文档，采用以下参数配置：

```bash
-c:v h264_bm          # 算能H.264硬件编码器
-b:v 1000k            # 视频码率
-r 10                 # 输出帧率
-g 30                 # GOP大小
-bf 0                 # B帧数量
-maxrate 1200k        # 最大码率
-bufsize 2000k        # 缓冲区大小
-profile:v main       # H.264 profile
-level 4.0            # H.264 level
-rtsp_transport tcp   # 使用TCP传输
-muxdelay 0.1         # 降低延迟
```

### 3. 输入格式优化
- **格式**: `image2pipe` + `mjpeg`
- **优势**: 
  - 兼容性好
  - 处理效率高
  - 适合实时流处理

## 文件说明

### 核心文件
- `simple_rtsp_processor.py` - 主要的RTSP处理程序
- `start_sophon_rtsp.sh` - 一键启动脚本
- `test_sophon_rtsp.py` - 算能FFmpeg测试脚本

### 配置参数
- **输入流**: `rtsp://192.168.1.172/video0`
- **输出流**: `rtsp://localhost:8554/pcb_detection`
- **模型**: `models/yolov5s_16848_f16.bmodel`
- **置信度**: 0.3

## 使用方法

### 1. 快速启动
```bash
cd /data/pcb
./start_sophon_rtsp.sh
```

### 2. 手动启动
```bash
# 启动MediaMTX服务
cd /data/mediamtx
./mediamtx &

# 测试算能FFmpeg
python3 test_sophon_rtsp.py

# 启动PCB检测推流
python3 simple_rtsp_processor.py \
    --input_rtsp rtsp://192.168.1.172/video0 \
    --output_rtsp rtsp://localhost:8554/pcb_detection \
    --bmodel models/yolov5s_16848_f16.bmodel \
    --conf_thresh 0.3
```

## 观看地址

### RTSP流
- **地址**: `rtsp://192.168.1.131:8554/pcb_detection`
- **工具**: VLC播放器、FFplay等

### HLS流
- **地址**: `http://192.168.1.131:8888/pcb_detection/index.m3u8`
- **工具**: 浏览器、VLC播放器

### WebRTC流
- **地址**: `http://192.168.1.131:8889/pcb_detection`
- **工具**: 支持WebRTC的浏览器

## 检测功能

### 缺陷类型
- **连锡**: 红色框标记
- **未知缺陷**: 绿色框标记

### 检测参数
- **置信度阈值**: 0.3
- **NMS阈值**: 0.5
- **最大检测数**: 1000

## 性能优化

### 1. 硬件加速
- 使用算能专用编码器 `h264_bm`
- 硬件加速图像处理
- 优化内存使用

### 2. 流处理优化
- 帧率控制 (10fps)
- 码率自适应
- TCP传输确保稳定性

### 3. 延迟优化
- 无缓冲模式
- 降低mux延迟
- 队列管理

## 故障排除

### 1. FFmpeg编码器问题
```bash
# 检查算能FFmpeg版本
ffmpeg -version

# 检查编码器支持
ffmpeg -encoders | grep h264_bm
```

### 2. 推流连接问题
- 检查MediaMTX服务状态
- 确认网络连接
- 验证端口开放

### 3. 检测效果问题
- 调整置信度阈值
- 检查模型文件
- 验证输入流质量

## 技术架构

```
输入RTSP流 -> SAIL解码 -> YOLOv5检测 -> 绘制结果 -> 算能FFmpeg编码 -> 输出RTSP流
     ↓              ↓           ↓            ↓              ↓
  摄像头流    → 硬件加速解码 → AI推理 → 结果可视化 → 硬件编码推流
```

## 参考文档
- [算能多媒体用户手册](https://doc.sophgo.com/bm1688_sdk-docs/v1.9/docs_latest_release/docs/multimedia/guide/guide/Multimedia_Guide_zh.html#sophgo-ffmpeg)
- SOPHGO FFmpeg编码器文档 