import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from address import save_path
from data.pro_data_set import DataSet

import os

from net.o_net import ONet
from net.p_net import PNet
from net.r_net import RNet


class Trainer:
    def __init__(self, data_path, net, save_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.net = net().to(self.device)
        self.save_path = save_path
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.dataSet = DataSet(data_path)
        self.dataLoader = DataLoader(self.dataSet, batch_size=512, shuffle=True, num_workers=4)

    def train(self, stop_value):
        # 接着训练
        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path))
        else:
            print(f"NO {self.save_path}")

        loss = 0

        while True:
            for i, (x, cls, offset) in enumerate(self.dataLoader):
                x, cls, offset = x.to(self.device), cls.to(self.device), offset.to(self.device)
                _cls_out, _offset_out = self.net(x)
                cls_out, offset_out = _cls_out.view(-1, 1), _offset_out.view(-1, 4)

                # 计算分类的损失 lt< gt> eq= le<= ge>= ne!=
                # part样本不参与分类损失计算, 输出正样本、负样本的布尔值
                # 标签：根据掩码选择标签的正样本、负样本，如[1., 0., 1., 0., 1..]
                # 输出：根据掩码选择输出的正样本、负样本，如[1.0000e+00, 5.5516e-08, 1.0000e+00,.]
                cls_mask = torch.lt(cls, 2)
                cls_target = torch.masked_select(cls, cls_mask)
                cls_out = torch.masked_select(cls_out, cls_mask)

                cls_loss = self.cls_loss_fn(cls_out, cls_target)

                # 计算bound损失
                # 负样本不参与计算， 输出正样本、部分样本的布尔值
                # 标签：根据掩码选择标签的正样本、部分样本，如[1., 0., 1., 0., 1.。]
                # 输出：根据掩码选择输出的正样本、部分样本， 如[1.0000e+00, 5.5516e-08, 1.0000e+00,.]
                offset_mask = torch.gt(cls, 0)
                offset_target = torch.masked_select(offset, offset_mask)
                offset_out = torch.masked_select(offset_out, offset_mask)

                offset_loss = self.offset_loss_fn(offset_out, offset_target)

                # 两个损失各自优化。谁也不影响谁
                loss = 0.5 * cls_loss + 0.5 * offset_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"loss:{loss.float()}, cls_loss:{cls_loss.float()}, offset_loss:{offset_loss.float()}")

            torch.save(self.net.state_dict(), self.save_path)
            print(f"{self.save_path}")

            if loss.float() < stop_value:
                break


def train_net(data_path, net, save_path, stop_value):
    path = "../param/"  # 存储训练结果
    if not os.path.exists(data_path):
        print("训练集不存在！")
    else:
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, save_path)
        train = Trainer(data_path, net, save_path)
        train.train(stop_value)


if __name__ == "__main__":
    train_net(save_path + r"\12", PNet, "p_net.pth", 0.008)
    train_net(save_path + r"\24", RNet, "r_net.pth", 0.001)
    train_net(save_path + r"\48", ONet, "o_net.pth", 0.0005)
