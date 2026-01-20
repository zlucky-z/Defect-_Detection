#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import sys
import numpy as np
import cv2

# PCB连锡检测类别
COCO_CLASSES = ('link','unknown')

COLORS = [
        [128, 128, 128],  # 背景色
        [255, 0, 0],      # 连锡 - 红色
        [0, 255, 0],      # 未知 - 绿色
    ]

def is_img(file_name):
    """judge the file is available image or not
    Args:
        file_name (str): input file name
    Returns:
        (bool) : whether the file is available image
    """
    fmt = os.path.splitext(file_name)[-1]
    if isinstance(fmt, str) and fmt.lower() in ['.jpg','.png','.jpeg','.bmp','.jpeg','.webp']:
        return True
    else:
        return False