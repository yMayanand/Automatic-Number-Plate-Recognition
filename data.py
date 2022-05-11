from numpy import object0
import torch
from torch.utils import data
import pandas as pd
import cv2
from utils import extract_bbox, resize_tfms, assign_cell

df = pd.read_csv('./metadata.csv')


# base dataset
class ANPR(data.Dataset):
    def __init__(self, tfms=None, resize=None):
        super().__init__()
        self.resize = resize
        self.tfms = tfms
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df['file_prefixes'][idx]
        im = cv2.imread(f"{path}.jpeg")
        if im is None:
            im = cv2.imread(f"{path}.jpg")
        if im is None:
            im = cv2.imread(f"{path}.png")
        if im is None:
            im = cv2.imread(f"{path}.JPG")
        if self.tfms is not None:
            im = self.tfms(im)
        bbox, size = extract_bbox(f"{path}.xml")
        bbox = list(bbox.values())
        if self.resize:
            im, bbox = resize_tfms(im, bbox, size,
                                     new_size=self.resize)
        bbox = torch.tensor(bbox)
        objectness = torch.tensor([1.])
        gt = torch.cat([bbox, objectness])
        return im, gt,  
