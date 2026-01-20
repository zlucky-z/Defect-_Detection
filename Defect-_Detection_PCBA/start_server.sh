#!/bin/bash
# PCB缺陷检测系统 - 完整启动脚本
# 集成Web服务器和MediaMTX流媒体服务器

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
WEB_PORT=8040
MEDIAMTX_PORT=8554
MEDIAMTX_WEB_PORT=8889
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="/tmp"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   PCB缺陷检测系统 - 启动脚本${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 检查Python环境
echo -e "${YELLOW}[1/4] 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3已安装: $(python3 --version)${NC}"

# 检查依赖
echo -e "${YELLOW}[2/4] 检查依赖...${NC}"
if [ -f "requirements.txt" ]; then
    echo "检查Python依赖..."
    pip3 list | grep -q "Flask" || echo -e "${YELLOW}⚠️  Flask未安装，请运行: pip3 install -r requirements.txt${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt不存在${NC}"
fi
echo -e "${GREEN}✅ 依赖检查完成${NC}"

# 检查端口占用
echo -e "${YELLOW}[3/4] 检查端口占用...${NC}"
check_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}⚠️  端口 $port ($name) 已被占用${NC}"
        return 1
    else
        echo -e "${GREEN}✅ 端口 $port ($name) 可用${NC}"
        return 0
    fi
}

check_port $WEB_PORT "Web服务器"
check_port $MEDIAMTX_PORT "RTSP服务"
check_port $MEDIAMTX_WEB_PORT "MediaMTX Web"

# 清理旧进程
echo -e "${YELLOW}[4/4] 清理旧进程...${NC}"

# 清理Flask进程
if [ -f "$PID_DIR/pcb_server.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/pcb_server.pid")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "停止旧的Web服务器 (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null
        sleep 1
    fi
    rm -f "$PID_DIR/pcb_server.pid"
fi

# 清理MediaMTX进程
if [ -f "$PID_DIR/mediamtx_pcb.pid" ]; then
    OLD_PID=$(cat "$PID_DIR/mediamtx_pcb.pid")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "停止旧的MediaMTX进程 (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null
        sleep 1
    fi
    rm -f "$PID_DIR/mediamtx_pcb.pid"
fi

echo -e "${GREEN}✅ 清理完成${NC}"
echo ""

# 启动Web服务器
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   启动Web服务器${NC}"
echo -e "${BLUE}=========================================${NC}"

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv_1.9.0/opencv-python

# 启动Flask应用
nohup python3 server.py > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_DIR/pcb_server.pid"

# 等待服务器启动
sleep 3

# 检查服务器是否成功启动
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Web服务器启动成功 (PID: $SERVER_PID)${NC}"
else
    echo -e "${RED}❌ Web服务器启动失败${NC}"
    echo "请查看日志: $LOG_DIR/server.log"
    exit 1
fi

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   系统启动完成${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}📡 访问地址：${NC}"
echo -e "   Web界面:  ${BLUE}http://localhost:$WEB_PORT${NC}"
echo -e "   默认账号: ${YELLOW}admin / admin123${NC}"
echo ""
echo -e "${GREEN}📝 日志文件：${NC}"
echo -e "   Web服务器: $LOG_DIR/server.log"
echo -e "   MediaMTX:  $LOG_DIR/mediamtx.log (运行时)"
echo -e "   RTSP检测:  $LOG_DIR/rtsp_detector.log (运行时)"
echo ""
echo -e "${GREEN}💡 使用说明：${NC}"
echo -e "   1. 打开Web界面登录系统"
echo -e "   2. 进入'实时监控'页面启动RTSP检测"
echo -e "   3. MediaMTX会自动启动并管理流媒体服务"
echo ""
echo -e "${GREEN}🛑 停止服务：${NC}"
echo -e "   运行: ${YELLOW}./stop_server.sh${NC}"
echo -e "   或按: ${YELLOW}Ctrl+C${NC} 然后手动清理"
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${YELLOW}系统运行中... 按 Ctrl+C 停止${NC}"
echo -e "${BLUE}=========================================${NC}"

# 定义清理函数
cleanup() {
    echo ""
    echo -e "${YELLOW}正在停止服务...${NC}"
    
    # 停止Web服务器
    if [ -f "$PID_DIR/pcb_server.pid" ]; then
        PID=$(cat "$PID_DIR/pcb_server.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "停止Web服务器 (PID: $PID)..."
            kill $PID 2>/dev/null
            sleep 1
            kill -9 $PID 2>/dev/null
        fi
        rm -f "$PID_DIR/pcb_server.pid"
    fi
    
    # 停止MediaMTX
    if [ -f "$PID_DIR/mediamtx_pcb.pid" ]; then
        PID=$(cat "$PID_DIR/mediamtx_pcb.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "停止MediaMTX (PID: $PID)..."
            kill $PID 2>/dev/null
            sleep 1
            kill -9 $PID 2>/dev/null
        fi
        rm -f "$PID_DIR/mediamtx_pcb.pid"
    fi
    
    # 停止RTSP检测器
    pkill -f "rtsp_output_detector_optimized.py" 2>/dev/null
    
    echo -e "${GREEN}✅ 服务已停止${NC}"
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 保持脚本运行并显示日志
tail -f "$LOG_DIR/server.log"

