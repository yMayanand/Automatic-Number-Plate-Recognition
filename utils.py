import xml.etree.ElementTree as ET
import torch
from torchvision import transforms

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
def resize_tfms(image, bbox, old_size, new_size=(224, 224)):
    image = transforms.Resize(size=new_size)(image)
    bbox = resize_bbox(bbox, new_size=new_size, old_size=old_size)
    bbox = normal2cxcywh(bbox, size=new_size)
    return image, bbox

# function to convert bbox co-ordinates
def normal2cxcywh(bbox, size):
    # image will in format : (xmin, ymin, xmax, ymax)
    w, h = size
    xmin, ymin, xmax, ymax = bbox
    xmin /= w
    xmax /= w
    ymin /= h
    ymax /= h

    i_w = xmax - xmin
    i_h = ymax - ymin

    cx = xmin + i_w / 2
    cy = ymin + i_h / 2

    return cx, cy, i_w, i_h


# function to assign bbox to a cell in final feature map
def assign_cell(bbox, grid_size=(7, 7)):
    cx, cy, _, _, _ = bbox
    w, h = grid_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_h = torch.tensor([1 / h]).to(device)
    base_w = torch.tensor(1 / w).to(device)
    px = torch.div(cx , base_w, rounding_mode='floor')
    py = torch.div(cy, base_h, rounding_mode='floor')
    px = int(px.item())
    py = int(py.item())
    if px >= w:
        px = w - 1

    if py >= h:
        py = h - 1
    return (px, py)

    




