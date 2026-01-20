#!/bin/bash
# PCB缺陷检测系统 - 集成检查脚本

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   PCB系统集成检查${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2 ${RED}(缺失: $1)${NC}"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $2"
        return 0
    else
        echo -e "${RED}❌${NC} $2 ${RED}(缺失: $1)${NC}"
        return 1
    fi
}

TOTAL=0
PASSED=0

echo -e "${YELLOW}核心文件检查:${NC}"
check_file "server.py" "Flask Web服务器" && ((PASSED++)); ((TOTAL++))
check_file "config.py" "配置文件" && ((PASSED++)); ((TOTAL++))
check_file "start_server.sh" "启动脚本" && ((PASSED++)); ((TOTAL++))
check_file "stop_server.sh" "停止脚本" && ((PASSED++)); ((TOTAL++))
echo ""

echo -e "${YELLOW}YOLOv5文件检查:${NC}"
check_file "python/yolov5_bmcv.py" "YOLOv5检测器" && ((PASSED++)); ((TOTAL++))
check_file "python/utils.py" "工具函数" && ((PASSED++)); ((TOTAL++))
echo ""

echo -e "${YELLOW}YOLOv8集成文件检查:${NC}"
check_file "python/rtsp_output_detector_optimized.py" "YOLOv8 RTSP检测器" && ((PASSED++)); ((TOTAL++))
check_file "python/postprocess_numpy.py" "YOLOv8后处理" && ((PASSED++)); ((TOTAL++))
echo ""

echo -e "${YELLOW}MediaMTX检查:${NC}"
check_dir "mediamtx" "MediaMTX目录" && ((PASSED++)); ((TOTAL++))
check_file "mediamtx/mediamtx" "MediaMTX可执行文件" && ((PASSED++)); ((TOTAL++))
if [ -f "mediamtx/mediamtx" ]; then
    if [ -x "mediamtx/mediamtx" ]; then
        echo -e "${GREEN}✅${NC} MediaMTX可执行权限"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠️${NC}  MediaMTX无执行权限 ${YELLOW}(需要: chmod +x mediamtx/mediamtx)${NC}"
    fi
    ((TOTAL++))
fi
echo ""

echo -e "${YELLOW}模型文件检查:${NC}"
check_file "models/yolov5s_16848_f16.bmodel" "YOLOv5模型" && ((PASSED++)); ((TOTAL++))
if [ -f "models/yolov8s_186_f16.bmodel" ]; then
    echo -e "${GREEN}✅${NC} YOLOv8模型"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠️${NC}  YOLOv8模型 ${YELLOW}(需要从pcb_yolov8复制)${NC}"
fi
((TOTAL++))
echo ""

echo -e "${YELLOW}模板文件检查:${NC}"
check_file "templates/dashboard.html" "仪表板页面" && ((PASSED++)); ((TOTAL++))
check_file "templates/detection.html" "批量检测页面" && ((PASSED++)); ((TOTAL++))
check_file "templates/rtsp_monitor.html" "实时监控页面" && ((PASSED++)); ((TOTAL++))
check_file "templates/history.html" "历史记录页面" && ((PASSED++)); ((TOTAL++))
check_file "templates/monitoring.html" "系统监控页面" && ((PASSED++)); ((TOTAL++))
check_file "templates/settings.html" "系统设置页面" && ((PASSED++)); ((TOTAL++))
echo ""

echo -e "${YELLOW}目录检查:${NC}"
check_dir "logs" "日志目录" && ((PASSED++)); ((TOTAL++))
check_dir "results" "结果目录" && ((PASSED++)); ((TOTAL++))
check_dir "uploads" "上传目录" && ((PASSED++)); ((TOTAL++))
echo ""

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   检查结果: ${PASSED}/${TOTAL}${NC}"
echo -e "${BLUE}=========================================${NC}"

if [ $PASSED -eq $TOTAL ]; then
    echo -e "${GREEN}✅ 所有检查通过！系统已正确集成${NC}"
    echo ""
    echo -e "${GREEN}可以启动系统:${NC}"
    echo -e "   ${YELLOW}./start_server.sh${NC}"
    exit 0
elif [ $PASSED -ge $((TOTAL * 3 / 4)) ]; then
    echo -e "${YELLOW}⚠️  大部分检查通过，但有一些项目需要注意${NC}"
    echo ""
    echo -e "${YELLOW}建议检查上述警告项目后再启动系统${NC}"
    exit 1
else
    echo -e "${RED}❌ 检查失败较多，请修复后再启动系统${NC}"
    exit 2
fi

