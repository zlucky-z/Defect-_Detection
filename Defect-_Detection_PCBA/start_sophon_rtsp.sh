#!/bin/bash

# 算能PCB缺陷检测RTSP推流启动脚本
# 根据算能FFmpeg文档优化

echo "=== 算能PCB缺陷检测RTSP推流系统 ==="
echo "时间: $(date)"
echo "设备: 算能BM1688"
echo ""

# 配置参数
INPUT_RTSP="rtsp://192.168.1.172/video0"
OUTPUT_RTSP="rtsp://localhost:8554/pcb_detection"
BMODEL="models/yolov5s_16848_f16.bmodel"
CONF_THRESH=0.3
DEV_ID=0

echo "配置参数:"
echo "  输入流: $INPUT_RTSP"
echo "  输出流: $OUTPUT_RTSP"
echo "  模型: $BMODEL"
echo "  置信度: $CONF_THRESH"
echo "  设备ID: $DEV_ID"
echo ""

# 检查MediaMTX服务
echo "1. 检查MediaMTX服务状态..."
if pgrep -f "mediamtx" > /dev/null; then
    echo "✓ MediaMTX服务正在运行"
else
    echo "✗ MediaMTX服务未运行，正在启动..."
    cd /data/mediamtx
    nohup ./mediamtx > /dev/null 2>&1 &
    sleep 3
    if pgrep -f "mediamtx" > /dev/null; then
        echo "✓ MediaMTX服务启动成功"
    else
        echo "✗ MediaMTX服务启动失败"
        exit 1
    fi
fi

# 检查模型文件
echo ""
echo "2. 检查模型文件..."
if [ -f "$BMODEL" ]; then
    echo "✓ 模型文件存在: $BMODEL"
else
    echo "✗ 模型文件不存在: $BMODEL"
    exit 1
fi

# 测试算能FFmpeg编码器
echo ""
echo "3. 测试算能FFmpeg编码器..."
python3 test_sophon_rtsp.py
if [ $? -eq 0 ]; then
    echo "✓ 算能FFmpeg编码器测试通过"
else
    echo "✗ 算能FFmpeg编码器测试失败"
    echo "请检查FFmpeg版本和h264_bm编码器支持"
    exit 1
fi

# 启动PCB检测推流
echo ""
echo "4. 启动PCB缺陷检测推流..."
echo "使用算能专用FFmpeg编码器: h264_bm"
echo ""
echo "观看地址:"
echo "  RTSP: rtsp://192.168.1.131:8554/pcb_detection"
echo "  HLS:  http://192.168.1.131:8888/pcb_detection/index.m3u8"
echo "  WebRTC: http://192.168.1.131:8889/pcb_detection"
echo ""
echo "按 Ctrl+C 停止推流"
echo ""

# 运行检测程序
python3 simple_rtsp_processor.py \
    --input_rtsp "$INPUT_RTSP" \
    --output_rtsp "$OUTPUT_RTSP" \
    --bmodel "$BMODEL" \
    --conf_thresh $CONF_THRESH \
    --dev_id $DEV_ID

echo ""
echo "PCB检测推流已停止" 