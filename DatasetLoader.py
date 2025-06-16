import cv2
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))

    def __getitem__(self, index):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        label_path = self.imgs_path[index]
        target = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        input_label = label_path.replace('target', 'input')
        input = cv2.imread(input_label, cv2.IMREAD_GRAYSCALE)

        target = trans(target)
        input = trans(input)

        return input, target

    def __len__(self):
        return len(self.imgs_path)
