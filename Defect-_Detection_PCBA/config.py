# -*- coding: utf-8 -*-
"""PCB缺陷检测系统配置文件"""

# YOLOv5批量检测配置
DETECTION_CONFIG = {
    'model_path': 'models/yolov5s_16848_f16.bmodel',
    'input_size': (720, 1280),
    'confidence_threshold': 0.001,
    'nms_threshold': 0.6,
    'device_id': 0,
    # 与 python/utils.py 中的 COCO_CLASSES 保持一致
    'class_names': [
        'link',      # 连锡
        'unknown'    # 未知缺陷
    ],
    'class_colors': [
        [255, 0, 0],    # 连锡 - 红色
        [0, 255, 0]     # 未知 - 绿色
    ]
}

# YOLOv8 RTSP实时检测配置
YOLOV8_CONFIG = {
    'model_path': 'models/yolov8s_186_f16.bmodel',
    'input_rtsp': 'rtsp://192.168.1.195/video0',
    'output_rtsp': 'rtsp://192.168.1.214:8554/detection',
    'device_id': 0,
    'conf_thresh': 0.3,
    'nms_thresh': 0.7,
    'frame_skip': 2,      # 还原：处理1帧跳过2帧（保持原有性能平衡）
    'output_fps': 15,     # 还原：保持15fps输出
    'low_latency_mode': True,
    'class_names': [
        'link',      # 连锡
        'unknown'    # 未知缺陷
    ]
}

# MediaMTX流媒体服务器配置
MEDIAMTX_CONFIG = {
    'binary_path': 'mediamtx/mediamtx',
    'pid_file': '/tmp/mediamtx_pcb.pid',
    'log_file': 'logs/mediamtx.log',
    'rtsp_port': 8554,
    'web_port': 8889,
    'auto_start': True
}

# 系统配置
SYSTEM_CONFIG = {
    'host': '0.0.0.0',
    'port': 8040,
    'debug': True,
    'upload_folder': './uploads',
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
}

# 用户配置
USERS_CONFIG = {
    'admin': {
        'password': 'admin123',
        'role': 'administrator',
        'name': '系统管理员'
    },
    'operator': {
        'password': 'operator123',
        'role': 'operator', 
        'name': '操作员'
    }
} 