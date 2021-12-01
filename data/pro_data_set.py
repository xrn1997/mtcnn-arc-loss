from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch

from address import save_path

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]
)

class DataSet(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataSet = []
        self.dataSet.extend(open(os.path.join(path, "positive.txt")).readlines())
        self.dataSet.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataSet.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataSet)

    def __getitem__(self, index):
        strs = self.dataSet[index].strip().split()
        img_path = os.path.join(self.path, strs[0])
        img_data = Image.open(img_path)
        img_data = transform(img_data)
        cond = torch.tensor([int(strs[1])], dtype=torch.float32)
        offsets = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])

        return img_data, cond, offsets


if __name__ == "__main__":
    # 这个r是表示不转义，使用真实字符
    dataSet = DataSet(save_path+r"\12")
    dataLoader = DataLoader(dataset=dataSet, batch_size=10, shuffle=True)
    for i, (x, cls, offset) in enumerate(dataLoader):
        print(x.shape)
        print(cls)
        print(offset)
        break
