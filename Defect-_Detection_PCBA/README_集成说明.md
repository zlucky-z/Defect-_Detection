# PCB缺陷检测系统 - YOLOv8集成版

## 🎉 集成完成

本系统已成功集成YOLOv8实时RTSP流检测功能，在保留原有YOLOv5批量检测功能的基础上，增加了高性能实时视频流检测能力。

## ✨ 功能特性

### 原有功能（YOLOv5）
- ✅ 用户登录与权限管理
- ✅ 批量图片检测
- ✅ 单张图片上传检测
- ✅ 检测历史记录
- ✅ 数据统计分析
- ✅ 系统监控

### 新增功能（YOLOv8）
- 🆕 **实时RTSP流检测**（22+ FPS）
- 🆕 **MediaMTX流媒体服务器集成**
- 🆕 **低延迟模式**（~30ms延迟）
- 🆕 **Web界面实时监控控制**
- 🆕 **智能跳帧优化**
- 🆕 **自动流管理**

## 🚀 快速启动

### 1. 启动完整系统

```bash
cd /data/pcb/det/pcb
./start_server.sh
```

系统会自动启动：
- Flask Web服务器（端口8040）
- 准备MediaMTX流媒体服务器（按需启动）

### 2. 访问Web界面

打开浏览器访问：
```
http://localhost:8040
```

默认登录账号：
- 管理员：`admin` / `admin123`
- 操作员：`operator` / `operator123`

### 3. 使用实时监控

1. 登录后点击左侧菜单 **"实时监控"**
2. 配置RTSP输入地址（默认：rtsp://192.168.1.194/video0）
3. 调整检测参数（置信度、NMS阈值）
4. 点击 **"启动检测"** 开始实时检测
5. 系统会自动启动MediaMTX并开始推流

### 4. 观看检测结果

有三种方式观看实时检测结果：

**方式1：VLC播放器**
```
打开VLC → 媒体 → 打开网络串流
输入：rtsp://localhost:8554/detection
```

**方式2：MediaMTX Web播放器**
```
浏览器打开：http://localhost:8889
```

**方式3：FFplay命令行**
```bash
ffplay rtsp://localhost:8554/detection
```

### 5. 停止系统

```bash
./stop_server.sh
```

## 📁 项目结构

```
/data/pcb/det/pcb/
├── server.py                    # Flask Web服务器（已集成RTSP管理）
├── config.py                    # 配置文件（已添加YOLOv8配置）
├── start_server.sh              # 启动脚本
├── stop_server.sh               # 停止脚本
├── python/
│   ├── yolov5_bmcv.py          # YOLOv5批量检测
│   ├── rtsp_output_detector_optimized.py  # YOLOv8实时检测（新增）
│   ├── postprocess_numpy.py    # YOLOv8后处理
│   └── utils.py                 # 工具函数
├── models/
│   ├── yolov5s_16848_f16.bmodel  # YOLOv5模型
│   └── yolov8s_186_f16.bmodel    # YOLOv8模型（需复制）
├── mediamtx/                    # MediaMTX流媒体服务器（新增）
│   ├── mediamtx                 # 服务器可执行文件
│   └── mediamtx.yml            # 配置文件
├── templates/
│   ├── dashboard.html           # 仪表板
│   ├── detection.html           # 批量检测页面
│   ├── rtsp_monitor.html        # 实时监控页面（新增）
│   ├── history.html             # 历史记录
│   ├── monitoring.html          # 系统监控
│   └── settings.html            # 系统设置
└── logs/                        # 日志目录
    ├── server.log               # Web服务器日志
    ├── mediamtx.log            # MediaMTX日志
    └── rtsp_detector.log       # RTSP检测器日志
```

## 🔧 配置说明

### YOLOv8实时检测配置（config.py）

```python
YOLOV8_CONFIG = {
    'model_path': 'models/yolov8s_186_f16.bmodel',  # YOLOv8模型路径
    'input_rtsp': 'rtsp://192.168.1.194/video0',    # 输入RTSP流
    'output_rtsp': 'rtsp://localhost:8554/detection', # 输出RTSP流
    'device_id': 0,                                   # 设备ID
    'conf_thresh': 0.3,                               # 置信度阈值
    'nms_thresh': 0.7,                                # NMS阈值
    'frame_skip': 2,                                  # 跳帧数（处理1帧跳2帧）
    'output_fps': 15,                                 # 输出帧率
    'low_latency_mode': True                          # 低延迟模式
}
```

### MediaMTX配置

```python
MEDIAMTX_CONFIG = {
    'binary_path': 'mediamtx/mediamtx',          # MediaMTX可执行文件
    'pid_file': '/tmp/mediamtx_pcb.pid',         # PID文件
    'log_file': 'logs/mediamtx.log',             # 日志文件
    'rtsp_port': 8554,                            # RTSP端口
    'web_port': 8889,                             # Web播放器端口
    'auto_start': True                            # 自动启动
}
```

## 🎯 检测类别

系统检测PCB上的两种缺陷：

| 类别ID | 类别名称 | 显示颜色 | 说明 |
|--------|----------|----------|------|
| 0 | link | 🔴 红色 | 连锡缺陷 |
| 1 | unknown | 🟢 绿色 | 未知缺陷 |

## 📊 性能指标

### YOLOv5批量检测
- 检测速度：根据图片数量和硬件
- 适用场景：批量离线检测

### YOLOv8实时检测
- 检测速度：22+ FPS
- 检测延迟：~30ms/帧
- 输入格式：RTSP流 (1920x1080)
- 输出格式：RTSP流 (1920x1080@15fps)
- 跳帧机制：处理1帧，跳过2帧（可配置）

## 🔌 API接口

### RTSP流管理API

**启动RTSP检测**
```http
POST /api/rtsp/start
Content-Type: application/json

{
  "input_rtsp": "rtsp://192.168.1.194/video0",
  "conf_thresh": 0.3,
  "nms_thresh": 0.7
}
```

**停止RTSP检测**
```http
POST /api/rtsp/stop
```

**获取RTSP状态**
```http
GET /api/rtsp/status
```

**获取RTSP配置**
```http
GET /api/rtsp/config
```

## 🐛 故障排除

### 1. MediaMTX启动失败

**检查端口占用：**
```bash
sudo netstat -tlnp | grep 8554
```

**查看日志：**
```bash
cat logs/mediamtx.log
```

### 2. RTSP连接失败

**测试输入流：**
```bash
ffprobe rtsp://192.168.1.194/video0
```

**测试输出流：**
```bash
ffplay rtsp://localhost:8554/detection
```

### 3. 检测器启动失败

**查看日志：**
```bash
tail -f logs/rtsp_detector.log
```

**检查模型文件：**
```bash
ls -lh models/yolov8s_186_f16.bmodel
```

### 4. Web界面无法访问

**检查服务器状态：**
```bash
ps aux | grep server.py
```

**查看日志：**
```bash
tail -f logs/server.log
```

## 📝 使用建议

### 批量检测场景
- 使用 **"检测管理"** 页面的YOLOv5批量检测
- 适合离线图片批量处理
- 可调整置信度和NMS阈值

### 实时监控场景
- 使用 **"实时监控"** 页面的YOLOv8实时检测
- 适合生产线实时监控
- 支持低延迟模式
- 可通过跳帧调节性能

## 🔄 版本信息

- **集成版本**: v2.0
- **集成日期**: 2026-01-19
- **基础版本**: YOLOv5 v1.x
- **新增功能**: YOLOv8实时RTSP流检测

## 💡 技术亮点

1. **双模型支持**: YOLOv5批量 + YOLOv8实时
2. **自动流管理**: MediaMTX按需启动停止
3. **Web统一管理**: 单一界面控制所有功能
4. **高性能优化**: 跳帧、低延迟、智能缓冲
5. **完美兼容**: 保留所有原有功能

## 📞 技术支持

如遇问题，请检查：
1. 硬件环境（Sophon SDK/BM1684X）
2. 网络连接（RTSP流地址）
3. 模型文件（是否存在且完整）
4. 日志文件（详细错误信息）

---

**🎯 开始使用**: `./start_server.sh`

