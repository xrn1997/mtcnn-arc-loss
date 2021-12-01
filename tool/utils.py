from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy
import numpy as np


def iou(box, boxes, isMin=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # [x1,y1,x2,y2,c]
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    inter_area = w * h
    if isMin:
        ratio = np.true_divide(inter_area, np.minimum(box_area, boxes_area))
    else:
        ratio = np.true_divide(inter_area, box_area + boxes_area - inter_area)
    return ratio


def nms(boxes, thresh=0.3, isMin=False):
    if boxes.shape[0] == 0:
        return np.array([])
    # 根据置信度排序
    boxes = boxes[(-boxes[:, 4]).argsort()]  # 不带-从小到大排序，带-从大到小排序。
    # 保留剩余的框
    _boxes = []

    while boxes.shape[0] > 1:
        # 取出第一个框
        max_box = boxes[0]
        other_boxes = boxes[1:]
        # 保留第一个框
        _boxes.append(max_box)

        # 比较iou后保留阈值小的值
        index = np.where(iou(max_box, other_boxes, isMin) < thresh)
        boxes = other_boxes[index]

    if boxes.shape[0] > 0:
        _boxes.append(boxes[0])
    return np.stack(_boxes)


def convert_to_square(boxes):
    square_boxes = boxes.copy()
    if boxes.shape[0] == 0:
        return np.array([])
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    max_side = np.maximum(w, h)
    square_boxes[:, 0] = boxes[:, 0] + w / 2 - max_side / 2
    square_boxes[:, 1] = boxes[:, 1] + h / 2 - max_side / 2
    square_boxes[:, 2] = square_boxes[:, 0] + max_side
    square_boxes[:, 3] = square_boxes[:, 1] + max_side

    return square_boxes


def trans_square(image):
    r"""Open the image using PIL."""
    image = image.convert('RGB')
    w, h = image.size
    background = Image.new('RGB', size=(max(w, h), max(w, h)), color=(0, 0, 0))  # 创建背景图，颜色值为127
    length = int(abs(w - h) // 2)  # 一侧需要填充的长度
    box = (length, 0) if w < h else (0, length)  # 粘贴的位置
    background.paste(image, box)
    # background.save("./merge2.jpg")

    return background


def npz2list(path):
    # 保存列表
    # numpy.savez('list1',list_in)
    list1 = np.load(path, allow_pickle=True)
    # print(list1.files) # 查看各个数组名称
    arr_0 = list1['arr_0']  # object 只读
    list_o = []
    for i in arr_0:
        list_o.append(i)
    return list_o


def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    a = np.array([1, 1, 11, 11])
    bs = np.array([[1, 1, 10, 10], [14, 15, 20, 20]])
    print(iou(a, bs))

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print((-bs[:, 4]).argsort())
    print(nms(bs))

    image = Image.open("../picture/pro_test_data/2.jpg")
    trans_square(image)
