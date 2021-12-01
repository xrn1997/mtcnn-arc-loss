import cv2

from data import pro_data_set
from net.face_net import *
from data.face_data_set import tf
from net.o_net import ONet
from net.p_net import PNet
from net.r_net import RNet
from tool.utils import nms, convert_to_square, cv2ImgAddText
from PIL import Image

import time
import numpy as np
import os


class Detector:
    def __init__(self, isCuda=True):
        self.isCuda = isCuda
        self.p_net = PNet()
        self.r_net = RNet()
        self.o_net = ONet()
        if self.isCuda:
            self.p_net.cuda()
            self.r_net.cuda()
            self.o_net.cuda()

        self.p_net.load_state_dict(torch.load("./param/p_net.pth"))
        self.r_net.load_state_dict(torch.load("./param/r_net.pth"))
        self.o_net.load_state_dict(torch.load("./param/o_net.pth"))

        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

        self.transform = pro_data_set.transform

    def detect(self, image):

        start_time = time.time()  # 获取当前时间的函数。
        pnet_boxes = self.__pnet_detect(image)
        if pnet_boxes.shape[0] == 0:  # 防止程序格式错误
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))
        return onet_boxes

    def __pnet_detect(self, image, stride=2, side=12):
        scale = 1  # 缩放比例
        boxes = []
        w, h = image.size
        min_side = min(w, h)

        while min_side >= 12:
            img_data = self.transform(image)
            img_data = img_data.unsqueeze(0)
            if self.isCuda:
                img_data = img_data.cuda()
            cls, offset = self.p_net(img_data)  # offset.shape=[1, 4, 528, 795]
            _cls, _offset = cls[0][0].cpu().data, offset[0].cpu().data  # _offset.shape=[4, 528, 795]
            mask = torch.gt(_cls, 0.6)  # 取置信度大于0.6的索引
            index = torch.nonzero(mask, as_tuple=False)  # (321,2)

            x1 = (index[:, 1] * stride) / scale  # (321)
            y1 = (index[:, 0] * stride) / scale
            x2 = x1 + side / scale
            y2 = y1 + side / scale
            _w = x2 - x1  # (321)
            _h = y2 - y1
            _x1 = _offset[0, index[:, 0], index[:, 1]] * _w + x1
            _y1 = _offset[1, index[:, 0], index[:, 1]] * _h + y1
            _x2 = _offset[2, index[:, 0], index[:, 1]] * _w + x2
            _y2 = _offset[3, index[:, 0], index[:, 1]] * _h + y2
            _cls = _cls[index[:, 0], index[:, 1]]
            box = torch.stack([_x1, _y1, _x2, _y2, _cls], dim=1)
            boxes.extend(box.numpy())
            scale *= 0.709  # 图像金字塔缩放比例0.3~0.7
            img_w, img_h = int(scale * w), int(scale * h)
            image = image.resize((img_w, img_h))
            min_side = min(img_w, img_h)

        return nms(np.array(boxes))

    def __rnet_detect(self, image, p_boxes):
        p_boxes = convert_to_square(p_boxes)
        x1 = p_boxes[:, 0]  # (668.)
        y1 = p_boxes[:, 1]
        x2 = p_boxes[:, 2]
        y2 = p_boxes[:, 3]
        box = np.stack((x1, y1, x2, y2), axis=1)
        img_dataset = [self.transform(image.crop(x).resize((24, 24))) for x in box]
        img_dataset = torch.stack(img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        cls, offset = self.r_net(img_dataset)
        cls, offset = cls.cpu().data.numpy(), offset.cpu().data.numpy()  # (668, 1) (668, 4)
        index, _ = np.where(cls > 0.6)  # (101,)
        box = p_boxes[index]  # (101, 5)
        x1 = box[:, 0]  # (101,)
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]
        w = x2 - x1  # (101,)
        h = y2 - y1
        _x1 = offset[index, 0] * w + x1
        _y1 = offset[index, 1] * h + y1
        _x2 = offset[index, 2] * w + x2
        _y2 = offset[index, 3] * h + y2
        _cls = cls[index, 0]
        boxes = np.stack((_x1, _y1, _x2, _y2, _cls), axis=1)
        return nms(boxes, isMin=False)

    def __onet_detect(self, image, r_boxes):
        r_boxes = convert_to_square(r_boxes)
        x1 = r_boxes[:, 0]
        y1 = r_boxes[:, 1]
        x2 = r_boxes[:, 2]
        y2 = r_boxes[:, 3]
        r_box = np.stack((x1, y1, x2, y2), axis=1)
        img_dataset = [self.transform(image.crop(x).resize((48, 48))) for x in r_box]
        img_dataset = torch.stack(img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        cls, offset = self.o_net(img_dataset)
        cls, offset = cls.cpu().data.numpy(), offset.cpu().data.numpy()
        index, _ = np.where(cls > 0.97)  # (44,)
        x1 = r_boxes[index, 0]  # (44,)
        y1 = r_boxes[index, 1]
        x2 = r_boxes[index, 2]
        y2 = r_boxes[index, 3]
        w = x2 - x1  # (44,)
        h = y2 - y1
        _x1 = offset[index, 0] * w + x1
        _y1 = offset[index, 1] * h + y1
        _x2 = offset[index, 2] * w + x2
        _y2 = offset[index, 3] * h + y2
        _cls = cls[index, 0]
        boxes = np.stack((_x1, _y1, _x2, _y2, _cls), axis=1)
        return nms(boxes, isMin=True)


class FaceDetector:
    def __init__(self):
        path = r"./picture/face_test_data2/"
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
            if siam > threshold and siam > max_threshold:
                max_threshold = siam
                max_threshold_feature = person_feature
        # print('----------完美分割线----------------')
        if max_threshold > 0:
            name = self.face_dict[max_threshold_feature]
            y = time.time()
            # print(y - x)
            return name, max_threshold.item()
        return '', '0.0'


# 图片人脸检测，使用pro_test_data文件夹,217行的image_file图片路径可以改
# if __name__ == "__main__":
#     x = time.time()  # 侦测开始计时
#     font = ImageFont.truetype(r"C:\Windows\Fonts\simhei", size=20)
#     with torch.no_grad() as grad:
#         image_file = "picture/pro_test_data/10.jpg"  # 图片路径
#         detector = Detector()  # 实例化
#         with Image.open(image_file) as img:
#             p_img = img.copy()
#             r_img = img.copy()
#             o_img = img.copy()
#             P_boxes, R_boxes, O_boxes = detector.detect(img)  # 将图片传入detect进行侦测，得到真实框的所有值
#             print(P_boxes.shape, R_boxes.shape, O_boxes.shape)
#             # imDraw = ImageDraw.Draw(p_img)  # 画P网络输出图
#             # for box in P_boxes:  # 遍历所有P网络输出的真实框 box[4] 为置信度
#             #     x1 = int(box[0])
#             #     y1 = int(box[1])
#             #     x2 = int(box[2])
#             #     y2 = int(box[3])
#             #     cls = box[4]
#             #     imDraw.rectangle((x1, y1, x2, y2), outline='red')  # 画出侦测后的所有真实框
#             # imDraw = ImageDraw.Draw(r_img)  # 画R网络输出图
#             # for box in R_boxes:  # 遍历所有R网络输出的真实框 box[4] 为置信度
#             #     x1 = int(box[0])
#             #     y1 = int(box[1])
#             #     x2 = int(box[2])
#             #     y2 = int(box[3])
#             #     cls = box[4]
#             #     imDraw.rectangle((x1, y1, x2, y2), outline='red')  # 画出侦测后的所有真实框
#             #     imDraw.text((x1, y1), "{:.3f}".format(cls), fill="red", font=font)
#             imDraw = ImageDraw.Draw(o_img)  # 画O网络输出图
#             for box in O_boxes:  # 遍历所有O网络输出的真实框 box[4] 为置信度
#                 x1 = int(box[0])
#                 y1 = int(box[1])
#                 x2 = int(box[2])
#                 y2 = int(box[3])
#                 cls = box[4]
#                 imDraw.rectangle((x1, y1, x2, y2), outline='red')  # 画出侦测后的所有真实框
#                 imDraw.text((x1, y1), "{:.3f}".format(cls), fill="red", font=font)
#             y = time.time()  # 计算侦测总用时
#             print(y - x)
#             o_img.show()

# 视频人脸识别,使用face_test_data2文件夹，其中该文件夹下的目录照片对应311行dicts中的名字value，请根据情况修改。
if __name__ == '__main__':

    with torch.no_grad() as grad:
        face_detector = FaceDetector()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 调取内置摄像头

        w = int(cap.get(3))  # 获取图片的宽度
        h = int(cap.get(4))  # 获取图片的高度
        print(w)  # 640
        print(h)  # 480

        font = cv2.FONT_HERSHEY_COMPLEX

        count = 0
        lists = []
        while True:
            a = time.time()
            ret, frame1 = cap.read()  # 该步所用时间0.0079

            # 将十六进制数据转成 二进制数据
            if cv2.waitKey(int(1)) & 0xFF == ord("q"):  # 视频在播放的过程中按键，循环会中断）。
                break
            elif ret == False:  # 视频播放完了，循环自动中断。
                break

            frame2 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            width, height = frame2.size  # 640  480
            # 将传给网络图片的变小，侦测速度会变快
            frame3 = frame2.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)

            # print("时间：", y-x)

            detector = Detector()
            boxes = detector.detect(frame3)

            # print("侦测出来的框：", boxes)

            count_b = 0
            value = ""
            for box in boxes:  # 遍历这一帧中有多少个人脸框

                w = int((box[2] - box[0]) / 0.1)  # 将缩小的图片以比例反算回原图大小
                h = int((box[3] - box[1]) / 0.1)
                x1 = int((box[0]) / 0.1 - 0.2 * w)
                y1 = int((box[1]) / 0.1)
                x2 = int((box[2]) / 0.1 - 0.1 * w)
                y2 = int((box[3]) / 0.1 - 0.2 * h)

                # print("侦测该张图片的目标置信度", box[4])
                cv2.rectangle(frame1, (x1, y1), (x2, y2), [0, 0, 255], 1)

                # 注意： 如果检测出来的图片比较暗，就需要用opencv进行处理，然后再传到识别网络进行识别。
                face_crop = frame2.crop((x1, y1, x2, y2))  # 将侦测到的人脸裁剪下来以便传到分类网络

                dicts = {"0": "徐荣楠", "1": "迪丽热巴", "2": "黄晓明", "3": "刘辉", "4": "周杰伦", "5": "吴京", "6": "张泽"}

                x = time.time()
                name, max_threshold = face_detector.face_detector(face_crop)  # 用时0.04s

                y = time.time()
                print(y - x)

                # 普通侦测
                if len(name) == 0:
                    pass
                else:
                    value = dicts[str(name)]
                    lists.append(value)

                    frame1 = cv2ImgAddText(frame1, value, x1, y1, (255, 0, 0), 40)

            count = count + 1  # 每过完一帧计数一次
            print("检测的帧数：" + str(count))
            if (len(lists) > 20 and ((lists.count(value) / len(lists)) > 0.5)):
                print("您是：" + value)
                break

            b = time.time()
            # print("侦测一张图片所用时间：", y - x)
            print("FPS:", 1 / (b - a))
            cv2.imshow("Detect", frame1)

        cap.release()  # 将视频关了
        cv2.destroyAllWindows()
