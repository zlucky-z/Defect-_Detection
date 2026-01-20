#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试算能FFmpeg RTSP推流功能
"""

import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_sophon_ffmpeg():
    """测试算能FFmpeg编码器"""
    logging.info("=== 测试算能FFmpeg编码器 ===")
    
    # 简化测试命令
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'lavfi',
        '-i', 'testsrc=duration=5:size=640x480:rate=10',
        '-c:v', 'h264_bm',
        '-b:v', '1000k',
        '-r', '10',
        '-f', 'rtsp',
        'rtsp://localhost:8554/test_stream'
    ]
    
    try:
        logging.info("启动测试推流...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待3秒
        time.sleep(3)
        
        if process.poll() is None:
            logging.info("✓ 算能FFmpeg编码器工作正常")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            logging.error(f"✗ 算能FFmpeg编码器测试失败")
            logging.error(f"stderr: {stderr.decode()}")
            return False
            
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
        return False

def check_mediamtx():
    """检查MediaMTX服务状态"""
    logging.info("=== 检查MediaMTX服务 ===")
    
    try:
        result = subprocess.run(['pgrep', '-f', 'mediamtx'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("✓ MediaMTX服务正在运行")
            return True
        else:
            logging.warning("✗ MediaMTX服务未运行")
            return False
    except Exception as e:
        logging.error(f"检查MediaMTX服务时出错: {e}")
        return False

def main():
    logging.info("开始测试算能RTSP推流系统...")
    
    # 1. 检查MediaMTX服务
    if not check_mediamtx():
        logging.error("请先启动MediaMTX服务")
        return
    
    # 2. 测试算能FFmpeg编码器
    if test_sophon_ffmpeg():
        logging.info("✓ 算能FFmpeg RTSP推流测试通过")
        logging.info("现在可以运行实际的PCB检测推流了")
    else:
        logging.error("✗ 算能FFmpeg RTSP推流测试失败")
        logging.error("请检查FFmpeg版本和编码器支持")

if __name__ == '__main__':
    main() 