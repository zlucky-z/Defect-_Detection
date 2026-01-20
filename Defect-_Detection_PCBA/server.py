#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB缺陷检测Web系统
- 用户登录与权限管理
- 实时检测仪表板
- 批量检测管理
- 检测历史记录
- 系统设置与监控
- 数据统计分析
"""
import os
import subprocess
import json
import glob
import time
import hashlib
import uuid
import logging
import psutil
import shutil
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)

# 读取配置
from config import DETECTION_CONFIG, SYSTEM_CONFIG, USERS_CONFIG, YOLOV8_CONFIG, MEDIAMTX_CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
RESULT_IMG_DIR = os.path.join(BASE_DIR, 'results', 'images')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')  # 批量检测临时目录
SINGLE_UPLOAD_DIR = os.path.join(BASE_DIR, 'single_upload')  # 单张检测专用目录

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = 'pcb_detection_system_2024'

# 用户数据库
USERS_DB = {}
for username, user_info in USERS_CONFIG.items():
    USERS_DB[username] = {
        'password': hashlib.md5(user_info['password'].encode()).hexdigest(),
        'role': user_info['role'],
        'name': user_info['name']
    }

# 系统统计
SYSTEM_STATS = {
    'total_inspections': 0,
    'defects_found': 0,
    'pass_rate': 100.0,
    'system_uptime': '99.8%',
    'processing_speed': 0.0
}

# 检测历史
DETECTION_HISTORY = []

# RTSP流检测状态
RTSP_DETECTOR_STATUS = {
    'running': False,
    'process': None,
    'mediamtx_process': None,
    'start_time': None,
    'error': None
}

# 系统监控日志
MONITORING_LOGS = []
MAX_LOG_ENTRIES = 200  # 最多保存200条日志

# 网络监控历史数据
NETWORK_HISTORY = {
    'last_bytes_sent': 0,
    'last_bytes_recv': 0,
    'last_update': time.time()
}

# ====================== 认证装饰器 ======================
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# ====================== 页面路由 ======================
@app.route('/')
@login_required
def index():
    return render_template('dashboard.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/detection')
@login_required
def detection():
    return render_template('detection.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/monitoring')
@login_required
def monitoring():
    return render_template('monitoring.html')

@app.route('/rtsp-monitor')
@login_required
def rtsp_monitor():
    return render_template('rtsp_monitor.html')

# ====================== 认证API ======================
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username in USERS_DB:
            hashed_password = hashlib.md5(password.encode()).hexdigest()
            if USERS_DB[username]['password'] == hashed_password:
                session['user_id'] = username
                session['user_role'] = USERS_DB[username]['role']
                session['user_name'] = USERS_DB[username]['name']
                return jsonify({
                    'success': True,
                    'message': '登录成功',
                    'user': {
                        'username': username,
                        'role': USERS_DB[username]['role'],
                        'name': USERS_DB[username]['name']
                    }
                })
        
        return jsonify({'success': False, 'message': '用户名或密码错误'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True, 'message': '已退出登录'})

@app.route('/api/user-info')
@login_required
def api_user_info():
    return jsonify({
        'success': True,
        'user': {
            'username': session['user_id'],
            'role': session['user_role'],
            'name': session['user_name']
        }
    })

# ====================== 检测API ======================
@app.route('/api/run_detect', methods=['POST'])
@login_required
def api_run_detect():
    """启动批量检测"""
    try:
        data = request.get_json(force=True)
        conf = float(data.get('conf_thresh', 0.3))
        nms = float(data.get('nms_thresh', 0.7))
        input_dir = data.get('input_dir', 'dates/images')

        # 生成检测ID
        task_id = str(uuid.uuid4())
        start_time = time.time()

        # 清理旧结果
        if os.path.exists(RESULT_IMG_DIR):
            for f in os.listdir(RESULT_IMG_DIR):
                os.remove(os.path.join(RESULT_IMG_DIR, f))

        cmd = [
            'python3', 'python/yolov5_bmcv.py',
            '--input', input_dir,
            '--bmodel', 'models/yolov5s_16848_f16.bmodel',
            '--dev_id', '0',
            '--conf_thresh', str(conf),
            '--nms_thresh', str(nms)
        ]
        
        # 执行检测
        proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            return jsonify({'success': False, 'message': proc.stderr}), 500

        # 收集结果
        imgs = sorted(os.listdir(RESULT_IMG_DIR)) if os.path.exists(RESULT_IMG_DIR) else []
        
        # 解析检测结果
        json_files = glob.glob(os.path.join(BASE_DIR, 'results', '*result.json'))
        detections = {}
        total_defects = 0
        
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            try:
                with open(latest_json, 'r', encoding='utf-8') as jf:
                    res_list = json.load(jf)
                for rec in res_list:
                    img = rec.get('image_name', '')
                    defects = []
                    for bbox in rec.get('bboxes', []):
                        cid = bbox.get('category_id', -1)
                        defects.append({
                            'category_id': cid,
                            'category': DETECTION_CONFIG['class_names'][cid] if 0 <= cid < len(DETECTION_CONFIG['class_names']) else str(cid),
                            'score': bbox.get('score', 0),
                            'bbox': bbox.get('bbox', [])
                        })
                    detections[img] = defects
                    total_defects += len(defects)
            except Exception as e:
                print('解析结果 JSON 失败:', e)

        # 更新统计
        processing_time = time.time() - start_time
        SYSTEM_STATS['total_inspections'] += len(imgs)
        SYSTEM_STATS['defects_found'] += total_defects
        if SYSTEM_STATS['total_inspections'] > 0:
            SYSTEM_STATS['pass_rate'] = round((1 - SYSTEM_STATS['defects_found'] / SYSTEM_STATS['total_inspections']) * 100, 2)
        SYSTEM_STATS['processing_speed'] = round(len(imgs) / processing_time if processing_time > 0 else 0, 2)

        # 添加到历史记录（包含详细结果）
        record = {
            'id': task_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': session['user_name'],
            'input_dir': input_dir,
            'total_images': len(imgs),
            'total_defects': total_defects,
            'processing_time': round(processing_time, 2),
            'conf_thresh': conf,
            'nms_thresh': nms,
            'status': 'completed',
            # 保存详细检测结果
            'images': imgs,
            'detections': detections
        }
        DETECTION_HISTORY.insert(0, record)
        
        # 添加监控日志
        if total_defects > 0:
            add_monitoring_log('warning', f'批量检测完成: 发现{total_defects}个缺陷 ({len(imgs)}张图片)')
        else:
            add_monitoring_log('info', f'批量检测完成: 全部合格 ({len(imgs)}张图片)')

        return jsonify({
            'success': True, 
            'task_id': task_id,
            'images': imgs, 
            'detections': detections,
            'summary': {
                'total_images': len(imgs),
                'total_defects': total_defects,
                'processing_time': round(processing_time, 2)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    """单张图片上传检测"""
    try:
        logging.info("收到上传检测请求")
        
        if 'file' not in request.files:
            logging.error("请求中没有文件")
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.error("文件名为空")
            return jsonify({'success': False, 'message': '文件名为空'}), 400
        
        logging.info(f"接收文件: {file.filename}, 大小: {len(file.read())} bytes")
        file.seek(0)  # 重置文件指针
        
        # 保存文件到单张检测专用目录
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(SINGLE_UPLOAD_DIR, filename)
        
        logging.info(f"保存文件到: {filepath}")
        file.save(filepath)
        logging.info(f"文件保存成功")
        
        # 执行单张图片检测
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 获取检测参数（可以从请求中获取，或使用默认值）
        conf = float(request.form.get('conf_thresh', 0.3))
        nms = float(request.form.get('nms_thresh', 0.7))
        
        # 清理旧结果
        if os.path.exists(RESULT_IMG_DIR):
            for f in os.listdir(RESULT_IMG_DIR):
                os.remove(os.path.join(RESULT_IMG_DIR, f))
        
        # 执行检测
        cmd = [
            'python3', 'python/yolov5_bmcv.py',
            '--input', filepath,
            '--bmodel', 'models/yolov5s_16848_f16.bmodel',
            '--dev_id', '0',
            '--conf_thresh', str(conf),
            '--nms_thresh', str(nms)
        ]
        
        logging.info(f"执行检测命令: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=120)
        
        if proc.returncode != 0:
            logging.error(f"检测失败: stdout={proc.stdout}, stderr={proc.stderr}")
            return jsonify({'success': False, 'message': f'检测失败: {proc.stderr}'}), 500
        
        logging.info(f"检测完成: {proc.stdout}")
        
        # 收集结果
        imgs = sorted(os.listdir(RESULT_IMG_DIR)) if os.path.exists(RESULT_IMG_DIR) else []
        
        # 解析检测结果
        json_files = glob.glob(os.path.join(BASE_DIR, 'results', '*result.json'))
        detections = {}
        total_defects = 0
        
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            try:
                with open(latest_json, 'r', encoding='utf-8') as jf:
                    res_list = json.load(jf)
                for rec in res_list:
                    img = rec.get('image_name', '')
                    defects = []
                    for bbox in rec.get('bboxes', []):
                        cid = bbox.get('category_id', -1)
                        defects.append({
                            'category_id': cid,
                            'category': DETECTION_CONFIG['class_names'][cid] if 0 <= cid < len(DETECTION_CONFIG['class_names']) else str(cid),
                            'score': bbox.get('score', 0),
                            'bbox': bbox.get('bbox', [])
                        })
                    detections[img] = defects
                    total_defects += len(defects)
            except Exception as e:
                print('解析结果 JSON 失败:', e)
        
        # 更新统计
        processing_time = time.time() - start_time
        SYSTEM_STATS['total_inspections'] += len(imgs)
        SYSTEM_STATS['defects_found'] += total_defects
        if SYSTEM_STATS['total_inspections'] > 0:
            SYSTEM_STATS['pass_rate'] = round((1 - SYSTEM_STATS['defects_found'] / SYSTEM_STATS['total_inspections']) * 100, 2)
        SYSTEM_STATS['processing_speed'] = round(len(imgs) / processing_time if processing_time > 0 else 0, 2)
        
        # 添加到历史记录
        record = {
            'id': task_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': session['user_name'],
            'input_dir': '单张上传',
            'total_images': len(imgs),
            'total_defects': total_defects,
            'processing_time': round(processing_time, 2),
            'conf_thresh': conf,
            'nms_thresh': nms,
            'status': 'completed',
            'images': imgs,
            'detections': detections
        }
        DETECTION_HISTORY.insert(0, record)
        
        # 添加监控日志
        if total_defects > 0:
            add_monitoring_log('warning', f'单张检测完成: 发现{total_defects}个缺陷')
        else:
            add_monitoring_log('info', '单张检测完成: 合格')
        
        logging.info(f"检测完成，返回结果: {len(imgs)}张图片, {total_defects}个缺陷")
        
        return jsonify({
            'success': True,
            'message': '上传并检测成功',
            'task_id': task_id,
            'filename': filename,
            'images': imgs,
            'detections': detections,
            'summary': {
                'total_images': len(imgs),
                'total_defects': total_defects,
                'processing_time': round(processing_time, 2)
            }
        })
    except subprocess.TimeoutExpired:
        logging.error("检测超时")
        return jsonify({'success': False, 'message': '检测超时，请稍后重试'}), 500
    except Exception as e:
        logging.error(f"上传检测异常: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

# ====================== 数据API ======================
@app.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    try:
        # 获取系统资源数据
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            'cpu_usage': round(cpu_percent, 1),
            'memory_usage': round(memory.percent, 1),
            'disk_usage': round(disk.percent, 1)
        }
        
        return jsonify({
            'success': True,
            'stats': SYSTEM_STATS,
            'system': system_info,
            'recent_detections': DETECTION_HISTORY[:5]
        })
    except Exception as e:
        logging.error(f'获取仪表板数据失败: {e}')
        # 即使系统资源获取失败，也返回基本数据
        return jsonify({
            'success': True,
            'stats': SYSTEM_STATS,
            'system': {
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0
            },
            'recent_detections': DETECTION_HISTORY[:5]
        })

@app.route('/api/detection-history')
@login_required
def api_detection_history():
    # 返回历史记录列表（不包含详细结果）
    history_list = []
    for record in DETECTION_HISTORY:
        summary = {
            'id': record['id'],
            'timestamp': record['timestamp'],
            'user': record['user'],
            'input_dir': record['input_dir'],
            'total_images': record['total_images'],
            'total_defects': record['total_defects'],
            'processing_time': record['processing_time'],
            'conf_thresh': record['conf_thresh'],
            'nms_thresh': record['nms_thresh'],
            'status': record['status']
        }
        history_list.append(summary)
    
    return jsonify({
        'success': True,
        'history': history_list
    })

@app.route('/api/detection-history/<task_id>')
@login_required
def api_detection_history_detail(task_id):
    """获取检测历史的详细信息"""
    for record in DETECTION_HISTORY:
        if record['id'] == task_id:
            return jsonify({
                'success': True,
                'record': record
            })
    
    return jsonify({
        'success': False,
        'message': '未找到该检测记录'
    }), 404

@app.route('/api/system-config')
@login_required
def api_system_config():
    return jsonify({
        'success': True,
        'config': {
            'detection': DETECTION_CONFIG,
            'system': SYSTEM_CONFIG
        }
    })

# ====================== RTSP流管理API ======================
def start_mediamtx():
    """启动MediaMTX流媒体服务器"""
    try:
        # 检查是否已经运行
        pid_file = os.path.join(BASE_DIR, MEDIAMTX_CONFIG['pid_file'])
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)  # 检查进程是否存在
                return True, pid, "MediaMTX已在运行"
            except:
                os.remove(pid_file)
        
        # 启动MediaMTX
        mediamtx_path = os.path.join(BASE_DIR, MEDIAMTX_CONFIG['binary_path'])
        log_dir = os.path.join(BASE_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(BASE_DIR, MEDIAMTX_CONFIG['log_file'])
        
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [mediamtx_path],
                cwd=os.path.dirname(mediamtx_path),
                stdout=log,
                stderr=log,
                start_new_session=True
            )
        
        # 保存PID
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        time.sleep(2)  # 等待启动
        
        # 检查是否成功启动
        if process.poll() is None:
            return True, process.pid, "MediaMTX启动成功"
        else:
            return False, None, "MediaMTX启动失败"
            
    except Exception as e:
        return False, None, f"启动MediaMTX失败: {str(e)}"

def stop_mediamtx():
    """停止MediaMTX流媒体服务器"""
    try:
        pid_file = os.path.join(BASE_DIR, MEDIAMTX_CONFIG['pid_file'])
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(1)
                try:
                    os.kill(pid, 9)  # SIGKILL
                except:
                    pass
            except:
                pass
            os.remove(pid_file)
        return True, "MediaMTX已停止"
    except Exception as e:
        return False, f"停止MediaMTX失败: {str(e)}"

def get_mediamtx_status():
    """获取MediaMTX状态"""
    try:
        pid_file = os.path.join(BASE_DIR, MEDIAMTX_CONFIG['pid_file'])
        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, 0)
                return True, pid
            except:
                os.remove(pid_file)
                return False, None
        return False, None
    except:
        return False, None

@app.route('/api/rtsp/start', methods=['POST'])
@login_required
def api_rtsp_start():
    """启动RTSP检测"""
    global RTSP_DETECTOR_STATUS
    
    try:
        if RTSP_DETECTOR_STATUS['running']:
            return jsonify({'success': False, 'message': 'RTSP检测已在运行中'})
        
        # 获取参数
        data = request.get_json() or {}
        input_rtsp = data.get('input_rtsp', YOLOV8_CONFIG['input_rtsp'])
        conf_thresh = float(data.get('conf_thresh', YOLOV8_CONFIG['conf_thresh']))
        nms_thresh = float(data.get('nms_thresh', YOLOV8_CONFIG['nms_thresh']))
        
        # 启动MediaMTX
        success, pid, msg = start_mediamtx()
        if not success:
            return jsonify({'success': False, 'message': msg})
        
        RTSP_DETECTOR_STATUS['mediamtx_process'] = pid
        
        # 等待MediaMTX完全启动
        time.sleep(3)
        
        # 启动YOLOv8检测器
        model_path = os.path.join(BASE_DIR, YOLOV8_CONFIG['model_path'])
        detector_script = os.path.join(BASE_DIR, 'python/rtsp_output_detector_optimized.py')
        output_rtsp = YOLOV8_CONFIG['output_rtsp']
        
        log_dir = os.path.join(BASE_DIR, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        detector_log = os.path.join(log_dir, 'rtsp_detector.log')
        
        cmd = [
            'python3', detector_script,
            '--model_path', model_path,
            '--input_rtsp', input_rtsp,
            '--output_rtsp', output_rtsp,
            '--device_id', str(YOLOV8_CONFIG['device_id']),
            '--conf_thresh', str(conf_thresh),
            '--nms_thresh', str(nms_thresh),
            '--frame_skip', str(YOLOV8_CONFIG['frame_skip']),
            '--output_fps', str(YOLOV8_CONFIG['output_fps'])
        ]
        
        if YOLOV8_CONFIG['low_latency_mode']:
            cmd.append('--low_latency_mode')
        
        with open(detector_log, 'w') as log:
            process = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=log,
                stderr=log,
                start_new_session=True
            )
        
        RTSP_DETECTOR_STATUS['process'] = process
        RTSP_DETECTOR_STATUS['running'] = True
        RTSP_DETECTOR_STATUS['start_time'] = time.time()  # 存储Unix时间戳（秒）
        RTSP_DETECTOR_STATUS['error'] = None
        
        # 添加监控日志
        add_monitoring_log('info', f'RTSP实时检测启动成功 ({input_rtsp})')
        
        return jsonify({
            'success': True,
            'message': 'RTSP检测启动成功',
            'output_url': output_rtsp,
            'web_url': f"http://localhost:{MEDIAMTX_CONFIG['web_port']}"
        })
        
    except Exception as e:
        RTSP_DETECTOR_STATUS['error'] = str(e)
        return jsonify({'success': False, 'message': f'启动失败: {str(e)}'})

@app.route('/api/rtsp/stop', methods=['POST'])
@login_required
def api_rtsp_stop():
    """停止RTSP检测"""
    global RTSP_DETECTOR_STATUS
    
    try:
        # 停止检测器
        if RTSP_DETECTOR_STATUS['process']:
            try:
                RTSP_DETECTOR_STATUS['process'].terminate()
                time.sleep(2)
                RTSP_DETECTOR_STATUS['process'].kill()
            except:
                pass
            RTSP_DETECTOR_STATUS['process'] = None
        
        # 停止MediaMTX
        stop_mediamtx()
        RTSP_DETECTOR_STATUS['mediamtx_process'] = None
        
        RTSP_DETECTOR_STATUS['running'] = False
        RTSP_DETECTOR_STATUS['start_time'] = None
        
        # 添加监控日志
        add_monitoring_log('info', 'RTSP实时检测已停止')
        
        return jsonify({'success': True, 'message': 'RTSP检测已停止'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'停止失败: {str(e)}'})

@app.route('/api/rtsp/status')
@login_required
def api_rtsp_status():
    """获取RTSP检测状态"""
    global RTSP_DETECTOR_STATUS
    
    # 检查进程是否真的在运行
    if RTSP_DETECTOR_STATUS['running'] and RTSP_DETECTOR_STATUS['process']:
        if RTSP_DETECTOR_STATUS['process'].poll() is not None:
            RTSP_DETECTOR_STATUS['running'] = False
            RTSP_DETECTOR_STATUS['error'] = '检测器进程已停止'
    
    # 检查MediaMTX状态
    mediamtx_running, mediamtx_pid = get_mediamtx_status()
    
    return jsonify({
        'success': True,
        'status': {
            'running': RTSP_DETECTOR_STATUS['running'],
            'start_time': RTSP_DETECTOR_STATUS['start_time'],
            'error': RTSP_DETECTOR_STATUS['error'],
            'mediamtx_running': mediamtx_running,
            'mediamtx_pid': mediamtx_pid,
            'output_url': YOLOV8_CONFIG['output_rtsp'] if RTSP_DETECTOR_STATUS['running'] else None,
            'web_url': f"http://localhost:{MEDIAMTX_CONFIG['web_port']}" if mediamtx_running else None
        }
    })

@app.route('/api/rtsp/config')
@login_required
def api_rtsp_config():
    """获取RTSP配置"""
    return jsonify({
        'success': True,
        'config': YOLOV8_CONFIG
    })

@app.route('/api/rtsp/stats')
@login_required
def api_rtsp_stats():
    """获取RTSP检测统计信息"""
    try:
        stats_file = os.path.join(BASE_DIR, 'logs/rtsp_stats.json')
        if os.path.exists(stats_file):
            # 检查文件是否是最近更新的（5秒内）
            file_age = time.time() - os.path.getmtime(stats_file)
            if file_age < 5:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                return jsonify({
                    'success': True,
                    'stats': stats
                })
        
        # 如果文件不存在或太旧，返回默认值
        return jsonify({
            'success': True,
            'stats': {
                'fps': 0,
                'detection_count': 0,
                'processed_frames': 0,
                'total_frames': 0,
                'last_update': 0
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

# ====================== 系统监控API ======================
def add_monitoring_log(log_type, message):
    """添加监控日志"""
    global MONITORING_LOGS
    log_entry = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': log_type,  # info, warning, error
        'message': message
    }
    MONITORING_LOGS.insert(0, log_entry)
    if len(MONITORING_LOGS) > MAX_LOG_ENTRIES:
        MONITORING_LOGS = MONITORING_LOGS[:MAX_LOG_ENTRIES]

@app.route('/api/monitoring/system')
@login_required
def api_monitoring_system():
    """获取系统状态(CPU、内存、磁盘使用率)"""
    try:
        # 获取CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # 获取内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 获取主磁盘使用率(根分区)
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # 判断系统状态
        status = 'normal'
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = 'error'
        elif cpu_percent > 75 or memory_percent > 80 or disk_percent > 80:
            status = 'warning'
        
        return jsonify({
            'success': True,
            'cpu': round(cpu_percent, 1),
            'memory': round(memory_percent, 1),
            'disk': round(disk_percent, 1),
            'status': status
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/monitoring/services')
@login_required
def api_monitoring_services():
    """获取服务状态"""
    try:
        services = []
        
        # Web服务器状态(当前进程)
        current_process = psutil.Process()
        web_uptime = datetime.now() - datetime.fromtimestamp(current_process.create_time())
        services.append({
            'name': 'Web服务器',
            'status': 'running',
            'uptime': format_uptime(web_uptime.total_seconds())
        })
        
        # 检测引擎状态(检查RTSP检测器)
        if RTSP_DETECTOR_STATUS['running'] and RTSP_DETECTOR_STATUS['process']:
            if RTSP_DETECTOR_STATUS['process'].poll() is None:
                detector_uptime = time.time() - RTSP_DETECTOR_STATUS['start_time']
                services.append({
                    'name': '检测引擎',
                    'status': 'running',
                    'uptime': format_uptime(detector_uptime)
                })
            else:
                services.append({
                    'name': '检测引擎',
                    'status': 'stopped',
                    'uptime': '--'
                })
        else:
            services.append({
                'name': '检测引擎',
                'status': 'stopped',
                'uptime': '--'
            })
        
        # MediaMTX状态
        mediamtx_running, mediamtx_pid = get_mediamtx_status()
        if mediamtx_running:
            try:
                mediamtx_proc = psutil.Process(mediamtx_pid)
                mediamtx_uptime = datetime.now() - datetime.fromtimestamp(mediamtx_proc.create_time())
                services.append({
                    'name': '流媒体服务',
                    'status': 'running',
                    'uptime': format_uptime(mediamtx_uptime.total_seconds())
                })
            except:
                services.append({
                    'name': '流媒体服务',
                    'status': 'stopped',
                    'uptime': '--'
                })
        else:
            services.append({
                'name': '流媒体服务',
                'status': 'stopped',
                'uptime': '--'
            })
        
        # 检查磁盘空间，如果超过75%显示警告
        disk = psutil.disk_usage('/')
        if disk.percent > 75:
            services.append({
                'name': '存储空间',
                'status': 'warning',
                'uptime': f'{disk.percent:.1f}% 已用'
            })
        
        return jsonify({
            'success': True,
            'services': services
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def format_uptime(seconds):
    """格式化运行时间"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"

@app.route('/api/monitoring/logs')
@login_required
def api_monitoring_logs():
    """获取系统日志"""
    try:
        limit = int(request.args.get('limit', 20))
        
        # 如果日志为空，生成一些初始日志
        if len(MONITORING_LOGS) == 0:
            add_monitoring_log('info', '系统监控服务已启动')
            add_monitoring_log('info', 'Web服务器运行正常')
            
            # 检查磁盘空间
            disk = psutil.disk_usage('/')
            if disk.percent > 80:
                add_monitoring_log('warning', f'磁盘空间使用率较高: {disk.percent:.1f}%')
            
            # 检查CPU温度
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current > 70:
                                add_monitoring_log('warning', f'CPU温度较高: {entry.current}°C')
            except:
                pass
        
        return jsonify({
            'success': True,
            'logs': MONITORING_LOGS[:limit]
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/monitoring/network')
@login_required
def api_monitoring_network():
    """获取网络监控数据"""
    try:
        global NETWORK_HISTORY
        
        # 获取网络IO统计
        net_io = psutil.net_io_counters()
        current_time = time.time()
        
        # 计算速度
        time_delta = current_time - NETWORK_HISTORY['last_update']
        if time_delta > 0 and NETWORK_HISTORY['last_bytes_sent'] > 0:
            download_speed = (net_io.bytes_recv - NETWORK_HISTORY['last_bytes_recv']) / time_delta / (1024 * 1024)  # MB/s
            upload_speed = (net_io.bytes_sent - NETWORK_HISTORY['last_bytes_sent']) / time_delta / (1024 * 1024)  # MB/s
        else:
            download_speed = 0
            upload_speed = 0
        
        # 更新历史数据
        NETWORK_HISTORY['last_bytes_sent'] = net_io.bytes_sent
        NETWORK_HISTORY['last_bytes_recv'] = net_io.bytes_recv
        NETWORK_HISTORY['last_update'] = current_time
        
        return jsonify({
            'success': True,
            'download_speed': max(0, round(download_speed, 2)),
            'upload_speed': max(0, round(upload_speed, 2))
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/monitoring/storage')
@login_required
def api_monitoring_storage():
    """获取存储监控数据"""
    try:
        disks = []
        
        # 获取所有分区
        partitions = psutil.disk_partitions()
        
        for partition in partitions:
            # 跳过一些特殊的文件系统
            if partition.fstype == '' or 'loop' in partition.device or 'snap' in partition.mountpoint:
                continue
            
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'name': partition.device,
                    'mountpoint': partition.mountpoint,
                    'percent': round(usage.percent, 1),
                    'used': format_bytes(usage.used),
                    'total': format_bytes(usage.total),
                    'free': format_bytes(usage.free),
                    'fstype': partition.fstype
                })
            except PermissionError:
                # 跳过没有权限访问的分区
                continue
        
        return jsonify({
            'success': True,
            'disks': disks
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def format_bytes(bytes_value):
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

@app.route('/api/monitoring/temperature')
@login_required
def api_monitoring_temperature():
    """获取设备温度"""
    try:
        temperature = 0
        
        # 尝试获取CPU温度
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # 尝试获取最常见的温度传感器
                for name in ['coretemp', 'k10temp', 'cpu_thermal', 'soc_thermal']:
                    if name in temps:
                        entries = temps[name]
                        if entries:
                            temperature = int(entries[0].current)
                            break
                
                # 如果没有找到，使用第一个可用的
                if temperature == 0:
                    for name, entries in temps.items():
                        if entries:
                            temperature = int(entries[0].current)
                            break
        except Exception as e:
            logging.warning(f"无法读取温度传感器: {e}")
            # 使用CPU使用率估算温度(仅供演示)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            temperature = int(30 + cpu_percent * 0.5)  # 简单估算
        
        return jsonify({
            'success': True,
            'temperature': temperature
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/monitoring/logs/clear', methods=['POST'])
@login_required
def api_monitoring_logs_clear():
    """清空监控日志"""
    try:
        global MONITORING_LOGS
        MONITORING_LOGS = []
        add_monitoring_log('info', '日志已清空')
        
        return jsonify({
            'success': True,
            'message': '日志已清空'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ====================== 静态文件 ======================
@app.route('/results/<path:filename>')
def get_result_image(filename):
    return send_from_directory(RESULT_IMG_DIR, filename)

@app.route('/uploads/<path:filename>')
def get_upload_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/single_upload/<path:filename>')
def get_single_upload_file(filename):
    return send_from_directory(SINGLE_UPLOAD_DIR, filename)

if __name__ == '__main__':
    # 创建必要目录
    os.makedirs(RESULT_IMG_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(SINGLE_UPLOAD_DIR, exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    app.run(host=SYSTEM_CONFIG['host'], port=SYSTEM_CONFIG['port'], debug=SYSTEM_CONFIG['debug']) 