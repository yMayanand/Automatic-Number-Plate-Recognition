import xml.etree.ElementTree as ET
import torch
from torchvision import transforms
import cv2
import math

# function to get bounding box from xml file


def extract_bbox(path):
    root = ET.parse(path).getroot()
    size = []
    bbox = {}
    for i in root.findall('size'):
        for j in i:
            size.append(int(j.text))
    for i in root.findall('object/bndbox'):
        for j in i:
            bbox.update({j.tag: int(j.text)})
    return bbox, size


# function to resize bbox
def resize_bbox(bbox, new_size=(224, 224), old_size=None):
    new_w, new_h = new_size
    assert old_size != None, 'old_size should not be None'
    old_w, old_h, ch = old_size
    w_ratio, h_ratio = new_w / old_w, new_h / old_h
    xmin, ymin, xmax, ymax = bbox
    xmin, xmax = xmin * w_ratio, xmax * w_ratio
    ymin, ymax = ymin * h_ratio, ymax * h_ratio
    return xmin, ymin, xmax, ymax

# resize transforms image and bbox

def pascal2yolo(bbox, size):
    # image will in format : tensor shape: (batch_size, 4) (xmin, ymin, xmax, ymax)
    bbox1 = bbox.clone().detach()

    w, h = size
    bbox1[:, 3] = bbox1[:, 3] - bbox1[:, 1]
    bbox1[:, 2] = bbox1[:, 2] - bbox1[:, 0]

    bbox1[:, 0] = bbox1[:, 0] + (bbox1[:, 2] / 2)
    bbox1[:, 1] = bbox1[:, 1] + (bbox1[:, 3] / 2)

    bbox1[:, [0, 2]] = bbox1[:, [0, 2]] / w
    bbox1[:, [1, 3]] = bbox1[:, [1, 3]] / h

    return bbox1


def yolo2pascal(bbox, size):
    # image will in format : tensor shape: (batch_size, 4) (xmin, ymin, xmax, ymax)
    bbox1 = bbox.clone().detach()
    w, h = size
    bbox1[:, [0, 2]] = bbox1[:, [0, 2]] * w
    bbox1[:, [1, 3]] = bbox1[:, [1, 3]] * h

    bbox1[:, 0] = bbox1[:, 0] - (bbox1[:, 2] / 2)
    bbox1[:, 1] = bbox1[:, 1] - (bbox1[:, 3] / 2)

    bbox1[:, 3] = bbox1[:, 1] + (bbox1[:, 3])
    bbox1[:, 2] = bbox1[:, 0] + (bbox1[:, 2])

    return bbox1


# function to assign bbox to a cell in final feature map
def assign_cell(bbox, grid_size=(7, 7)):
    cx, cy, _, _, _ = bbox
    w, h = grid_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_h = torch.tensor([1 / h]).to(device)
    base_w = torch.tensor([1 / w]).to(device)
    px = torch.div(cx, base_w, rounding_mode='floor')
    py = torch.div(cy, base_h, rounding_mode='floor')
    px = int(px.item())
    py = int(py.item())
    if px >= w:
        px = w - 1

    if py >= h:
        py = h - 1
    return (py, px)


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_wh(old_size, new_size):
    w, h = old_size
    aspect_ratio = w / h
    if w < h:
        w_ = new_size
        h_ = w_ / aspect_ratio
    else:
        h_ = new_size
        w_ = h_ * aspect_ratio
    return int(w_), int(h_)


def yolo2coco(bbox, size):
    w, h = size
    bbox[:, 0] = bbox[:, 0] - (bbox[:, 2] / 2)
    bbox[:, 1] = bbox[:, 1] - (bbox[:, 3] / 2)

    bbox[:, [0, 2]] = bbox[:, [0, 2]] * w
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * h
    return bbox

# generating anchor box shape: 49*4 wh


def generate_boxes(dim):
    a = [1/1, 2/1, 1/2, 3/1]
    scale = [0.1, 0.15, 0.25, 0.35]
    prior_boxes = []
    for k, ratio in enumerate(a):
        for i in range(dim):
            for j in range(dim):
                cx = (j + 0.5) / dim
                cy = (i + 0.5) / dim
                prior_boxes.append(
                    [cx, cy, scale[k] * math.sqrt(ratio), scale[k] / math.sqrt(ratio)])

    return torch.tensor(prior_boxes)


# This function is from https://github.com/kuangliu/pytorch-ssd.
def iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )

    rb = torch.min(
        # [N,2] -> [N,1,2] -> [N,M,2]
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        # [M,2] -> [1,M,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou
