import os
import shutil
import time

import torch

from data.face_data_set import tf
from detect import Detector
from net.face_net import *
from PIL import Image


class FaceDetector:
    def __init__(self):
        path = r"./picture/face_test_data"
        net_path = "./param/face_net.pth"
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.net = FaceNet().to(device)
        self.net.load_state_dict(torch.load(net_path))
        self.net.eval()
        self.face_dict = {}

        for face_dir in os.listdir(path):
            for face_filename in os.listdir(os.path.join(path, face_dir)):
                person_path = os.path.join(path, face_dir, face_filename)
                img = Image.open(person_path)
                img = img.convert("RGB")
                img = img.resize((112, 112))
                person1 = tf(img).to(device)
                person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
                self.face_dict[person1_feature] = face_dir

    def face_detector(self, img):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        max_threshold = 0
        threshold = 0.7
        max_threshold_feature = 0
        person1 = tf(img).to(device)
        person1_feature = self.net.encode(torch.unsqueeze(person1, 0))
        kys = self.face_dict.keys()
        kys = list(kys)

        for person_feature in kys:
            # print(person_feature.shape)
            siam = compare(person1_feature, person_feature)
            # print(self.face_dict[person_feature], siam)
            if siam > max_threshold:
                max_threshold = siam
                max_threshold_feature = person_feature
        # print('----------完美分割线----------------')
        if max_threshold > 0:
            name = self.face_dict[max_threshold_feature]
            y = time.time()
            # print(y - x)
            return name, max_threshold.item()
        return '', '0.0'


# ########################################muct数据库预处理###################################
# # 一次性代码 将已知分类中的图片按个人用文件夹区分开
# def classification(person_num, count, path, i=0):
#     sum = person_num + i
#     while (i < sum):
#         j = 0
#         for n in os.listdir(path + r"/已知类别"):
#             # if not os.path.exists(path + r"/" + n[0:4]):
#             #     os.mkdir(path + r"/" + n[0:4])
#             if not os.path.exists(path + r"/" + str(i)):
#                 os.mkdir(path + r"/" + str(i))
#             if j < count:
#                 # shutil.move(path + r"/已知类别/" + n, path + r"/" + n[0:4])
#                 shutil.move(path + r"/已知类别/" + n, path + r"/" + str(i))
#             j = j + 1
#         i = i + 1
#
#
# # 去掉待分类中的图片，确保测试集中的图片没有在训练集和验证集中出现
# if __name__ == '__main__':
#     path1 = r"./picture/待分类"  # 测试集目录
#     path2 = r"./picture/face_data"  # 训练集目录
#     path3 = r"./picture/face_data2"  # 验证集目录
#     path4 = r"./picture/face_test_data"  # 测试集录入信息目录
#     for n in os.listdir(path1):
#         for m in os.listdir(path2 + r"/已知类别"):
#             if n == m:
#                 os.remove(path2 + r"/已知类别" + r"/" + n)
#         for m in os.listdir(path3 + r"/已知类别"):
#             if n == m:
#                 os.remove(path3 + r"/已知类别" + r"/" + n)
#     classification(199, 8, path2)  # 使用classification中的str(i)相关代码
#     classification(77, 5, path2, 199)
#
#     classification(199, 6, path3)
#     classification(77, 4, path3, 199)
#
#     # classification(199, 14, path4) #使用classification中的n[0:4]相关代码
#     # classification(77, 9, path4)
#
# #####################################################################################
# 验证未分类数据集
if __name__ == '__main__':
    path = r"./picture/待分类"
    with torch.no_grad() as grad:
        face_detector = FaceDetector()
        count = 0
        correct_count = 0
        lists = []
        times = 0
        for n in os.listdir(path):
            image_file = path + r"/" + n
            print(image_file)
            with Image.open(image_file) as img:
                width, height = img.size  # 640  480
                # 将传给网络图片的变小，侦测速度会变快
                img2 = img.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
                detector = Detector()
                boxes = detector.detect(img2)
                count_b = 0
                value = ""
                for box in boxes:  # 遍历这一帧中有多少个人脸框
                    x = time.time()
                    name, max_threshold = face_detector.face_detector(img2)  # 用时0.04s
                    y = time.time()
                    # print(y - x)
                    print("识别对象:", n[0:4])
                    print("识别结果:", name)
                    # 普通侦测
                    if len(name) == 0:
                        print("识别错误")
                        pass
                    elif n[0:4] == name:
                        print("识别正确")
                        correct_count = correct_count + 1
                    else:
                        print("识别错误")
                count = count + 1  # 每过完一帧计数一次
                print("侦测一张图片所用时间：", y - x)
                times = times + (y - x)
    print("正确率：" + str(correct_count / count * 100) + "%")
    print("总花费时间：" + str(times))
