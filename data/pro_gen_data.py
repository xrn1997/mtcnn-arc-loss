import os

import numpy as np
from PIL import Image
from tool import utils
import traceback

from address import save_path, anno_src, img_dir

for face_size in [12, 24, 48]:
    # for face_size in [12]:
    # for face_size in [24]:
    # for face_size in [48]:
    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.split()

                image_filename = strs[0].strip()
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)

                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    # 排除过小的标签
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]

                    # 计算出人脸中心点的位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    side_len = max(w, h)
                    # 生成随机数种子
                    float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
                    seed = float_num[np.random.randint(0, len(float_num))]  # 0.1-0.9

                    # 使正样本和部分样本数量翻倍
                    for _ in range(5):
                        # 让人脸中心点有少许的偏移
                        _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))
                        _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                        # 让人脸形成正方形，并且让坐标也有少许的偏离
                        _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))

                        _x1 = _cx - _side_len / 2
                        _y1 = _cy - _side_len / 2
                        _x2 = _x1 + _side_len
                        _y2 = _y1 + _side_len

                        if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                            continue

                        crop_box = [_x1, _y1, _x2, _y2]

                        # 计算坐标的偏移量
                        # 真实框x坐标减去偏移框的坐标再除以偏移框的边长
                        offset_x1 = (x1 - _x1) / _side_len
                        offset_y1 = (y1 - _y1) / _side_len
                        offset_x2 = (x2 - _x2) / _side_len
                        offset_y2 = (y2 - _y2) / _side_len

                        offset_px1 = 0  # (px1 - _x1) / side_len  # 五个关键点都只对一个点进行偏移
                        offset_py1 = 0  # (py1 - y1_) / side_len
                        offset_px2 = 0  # (px2 - x1_) / side_len
                        offset_py2 = 0  # (py2 - y1_) / side_len
                        offset_px3 = 0  # (px3 - x1_) / side_len
                        offset_py3 = 0  # (py3 - y1_) / side_len
                        offset_px4 = 0  # (px4 - x1_) / side_len
                        offset_py4 = 0  # (py4 - y1_) / side_len
                        offset_px5 = 0  # (px5 - x1_) / side_len
                        offset_py5 = 0  # (py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size))

                        # 取出标量
                        iou = utils.iou(crop_box, np.array(boxes))[0]

                        if iou > 0.65:  # 正样本
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    positive_count, 1, offset_x1, offset_y1,
                                    offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            positive_anno_file.flush()  # 释放内存
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif 0.6 > iou > 0.4:  # 部分样本
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2,
                                    offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                    offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.05:  # 负样本
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
                    # 负样本增强
                    _boxes = np.array(boxes)

                    for i in range(3):
                        side_len = np.random.randint(face_size, min(img_w, img_h))
                        x_ = np.random.randint(0, img_w - 0.9 * side_len)
                        y_ = np.random.randint(0, img_h - 0.9 * side_len)
                        if x_ < 0 or y_ < 0 or (x_ + side_len) > img_w or (y_ + side_len) > img_h:
                            continue

                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        # 裁剪框和标签框作比较
                        if np.max(utils.iou(crop_box, _boxes)) < 0.01:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(
                                negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1
            except:
                traceback.print_exc()
    except:
        traceback.print_exc()
