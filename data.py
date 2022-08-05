import os
from selectors import EpollSelector
import cv2
import albumentations as A
from utils import *
from torch.utils import data

# base dataset
class ANPR(data.Dataset):
    def __init__(self, root, df, resize=None):
        super().__init__()
        self.resize = resize
        self.df = df
        self.df['img_file'] = df['img_file'].apply(
            lambda x: os.path.join(root, x))

        self.clean_()

    def __len__(self):
        return len(self.df)

    def clean_(self):
        print('cleaning started...')
        for i, file in enumerate(self.df.img_file):
            if not os.path.exists(file):
                self.df.drop(index=i, inplace=True)

        self.df.reset_index(drop=True, inplace=True)
        print('finished cleaning')

    def __getitem__(self, idx):
        image = read_image(self.df.loc[idx, 'img_file'])
        w, h, c, xmin, ymin, xmax, ymax = self.df.iloc[idx, :7]
        size = (w, h)
        bbox = [(xmin, ymin, xmax, ymax)]
        # bbox = pascal2yolo(bbox, size)

        if self.resize is not None:
            transform = A.Compose([
                A.Resize(*get_wh(size, self.resize)[::-1], interpolation=cv2.INTER_AREA),
                A.CenterCrop(height=self.resize, width=self.resize),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

            t = transform(image=image, bboxes=bbox, class_labels=['number_plate'])
            t['bboxes'] = torch.tensor(t['bboxes'])
        else:
            t = {'image': image, 'bboxes': torch.tensor(bbox)}

        priors = generate_boxes(7)
        priors = torch.clamp(priors, min=0., max=1.)
        n = priors.shape[0]

        if t['bboxes'].numel():
            bbox = t['bboxes']
        else:
            bbox = torch.tensor([[0., 0., 0., 0.]])

        size_ = (self.resize, self.resize) or size
        ious = iou(yolo2pascal(priors, size_), bbox)
        max_val, idxs = torch.max(ious, dim=0)
        priors_ = torch.zeros_like(priors)
        priors_[idxs, :] = priors[idxs, :]
        targets = torch.zeros_like(priors)
        targets[idxs] = pascal2yolo(torch.tensor(bbox), size_).float()
        targets[:, :2] -= priors_[:, :2]
        clabel = torch.zeros(n, 1)
        if t['bboxes'].numel():
            clabel[idxs, :] = torch.tensor([1]).float()
        return t['image'], targets, clabel
