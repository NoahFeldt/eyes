import os
import torch
from torch.utils.data import Dataset
import cv2

class EyesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        classes = os.listdir(self.root_dir)
        self.class_lengths = []

        for c in classes:
            self.class_lengths.append(len(os.listdir(os.path.join(self.root_dir, c))))

        self.limmits = []
        for i in range(0, len(self.class_lengths)):
            self.limmits.append(sum(self.class_lengths[0:i]))
        #print(self.limmits)

    def __len__(self):
        return sum(self.class_lengths)

    def __getitem__(self, index):
        folder = divmod(index, 1000)
        img_path = os.path.join(self.root_dir, str(folder[0]), str(folder[1]) + ".png")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        y_lable = torch.tensor(folder[0])

        if self.transform:
            image = self.transform(image)

        return (image, y_lable)
