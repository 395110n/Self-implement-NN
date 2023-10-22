import os
import pandas as pd
import torchvision
import torch 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from d2l import torch as d2l
from torch.utils.data import DataLoader


img = matplotlib.image.imread("C:\\Users\\User\\deepLearning\\lmdl\\data\\catdog.jpg")

def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis = -1)
    return boxes

def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

def bbox_to_rect(bbox, color):
    return patches.Rectangle(xy=(bbox[0], bbox[1]), 
                             width=(bbox[2]-bbox[0]), height=(bbox[3]-bbox[1]), fill=False,
                             edgecolor=color, linewidth=2)

"""fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))"""

def read_data_bananas(is_train=True):
    data_dir = "C:\\Users\\User\\deepLearning\\lmdl\\data\\banana-detection"
    csv_frame = os.path.join(data_dir, 
                             "bananas_train" if is_train else "bananas_val","label.csv")
    
    csv_data = pd.read_csv(csv_frame)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        
        images.append(
            torchvision.io.read_image(
                os.path.join(data_dir, "bananas_train" if is_train else "bananas_val",
                              "images", img_name)
            )
        )
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1)/256


class BananasDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f" training examples" if is_train else " validation examples"))
    
    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])
    
    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    train_iter = DataLoader(BananasDataset(is_train=True), batch_size, shuffle=True)
    valid_iter = DataLoader(BananasDataset(is_train=False), batch_size)
    return train_iter, valid_iter


batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))

imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in (zip(axes, batch[1][0:10])):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors = ["w"])

d2l.plt.show()