#!/bin/bash
# PCB缺陷检测系统 - 停止脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PID_DIR="/tmp"

echo -e "${YELLOW}正在停止PCB检测系统...${NC}"

# 停止Web服务器
if [ -f "$PID_DIR/pcb_server.pid" ]; then
    PID=$(cat "$PID_DIR/pcb_server.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止Web服务器 (PID: $PID)..."
        kill $PID 2>/dev/null
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        echo -e "${GREEN}✅ Web服务器已停止${NC}"
    else
        echo "Web服务器未运行"
    fi
    rm -f "$PID_DIR/pcb_server.pid"
else
    echo "未找到Web服务器PID文件"
fi

# 停止MediaMTX
if [ -f "$PID_DIR/mediamtx_pcb.pid" ]; then
    PID=$(cat "$PID_DIR/mediamtx_pcb.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止MediaMTX (PID: $PID)..."
        kill $PID 2>/dev/null
        sleep 2
        if ps -p $PID > /dev/null 2>&1; then
            kill -9 $PID 2>/dev/null
        fi
        echo -e "${GREEN}✅ MediaMTX已停止${NC}"
    else
        echo "MediaMTX未运行"
    fi
    rm -f "$PID_DIR/mediamtx_pcb.pid"
else
    echo "未找到MediaMTX PID文件"
fi

# 停止RTSP检测器
echo "检查RTSP检测器进程..."
if pkill -f "rtsp_output_detector_optimized.py" 2>/dev/null; then
    echo -e "${GREEN}✅ RTSP检测器已停止${NC}"
else
    echo "RTSP检测器未运行"
fi

# 清理可能残留的mediamtx进程
pkill -f "mediamtx/mediamtx" 2>/dev/null

echo -e "${GREEN}✅ 所有服务已停止${NC}"

