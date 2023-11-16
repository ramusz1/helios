import numpy as np
import cv2
from PIL import Image
from .HyperTools import X2Cube
import os
from pathlib import Path


#### MANAGING RECTANGLES

def load_rects(path):
    rects =  np.loadtxt(path, dtype=int)
    # rects = [None if r[2] <= 0 or r[3] <= 0 else r for r in rects ]
    return rects

def save_rects(path, rects):
    # rects = [[0, 0, 0, 0] if x is None else x for x in rects]
    return np.savetxt(path, rects, fmt="%d", delimiter="\t")

def draw_rect(img, rect, color=(0,255,0), thickness = 1):
    if rect is None:
        return img.copy()
    x,y,w,h = rect
    return cv2.rectangle(img.copy(), (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

def center_rect(r, inverse=False):
    x,y,w,h = r
    if inverse:
        return np.array([x - 0.5 * w, y - 0.5 * h, w, h])
    return np.array([x + 0.5 * w, y + 0.5 * h, w, h])

def center_rects(r, inverse=False):
    x,y,w,h = np.split(r, 4, axis=-1)
    if inverse:
        return np.stack([x - 0.5 * w, y - 0.5 * h, w, h], axis=-1)
    return np.stack([x + 0.5 * w, y + 0.5 * h, w, h], axis=-1)

def bbox_out_of_bounds(bbox, H, W):
    return bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > W or bbox[1] + bbox[3] > H 

def bbox_completely_out_of_bounds(bbox, H, W):
    x1,y1,w,h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return x2 < 0 or y2 < 0 or x1 >= W or y1 >= H

#### LOADING IMAGES

def load_hsi(hsi_path, camera_type):
    if camera_type == 'HSI-VIS':
        n_bands = 16
        a = 4
    elif camera_type == 'HSI-RedNIR':
        n_bands = 15
        a = 4
    elif camera_type == 'HSI-NIR':
        n_bands = 25
        a = 5
    return X2Cube(np.array(Image.open(hsi_path)), B=[a,a], skip=[a,a], bandNumber=n_bands)

def load_rgb(rgb_path):
    return np.array(Image.open(rgb_path))


class ImageLoader:
    
    def __init__(self, path, camera_type):
        self.camera_type = camera_type
        self.is_hsi = camera_type.find("FalseColor") == -1
        self.path = path
        self.imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(path)),
            key=lambda x: int(x.split('.')[0]))
        # self.imgs = self.imgs[frame_skip:]
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        img = os.path.join(self.path, self.imgs[i])
        if self.is_hsi:
            return load_hsi(img, self.camera_type)
        else:
            return load_rgb(img)


def load_tags(description_path):
    with open(description_path, 'r') as f:
        tags = [x.strip() for x in f.read().splitlines() if len(x) > 0]
    
    # perform additional mapping on tags. For example I've noticed that there is a 'Scale Change' and 'Scale Variation'. Those two can be unified
    mapping = {'Scale Change': 'Scale Variation'}
    tags = list(set([mapping[t] if t in mapping else t for t in tags]))
    return tags


CAMERA_TYPES = ["HSI-VIS", "HSI-NIR", "HSI-RedNIR"]


class Scene:
    
    def __init__(self, dataset_path, camera_type, scene_name):
        assert camera_type in CAMERA_TYPES
        self.camera_type = camera_type
        input_path = os.path.join(dataset_path, camera_type, scene_name)
        input_path_falsecolor = os.path.join(dataset_path, camera_type + "-FalseColor", scene_name)
        y_true_path = os.path.join(input_path, "groundtruth_rect.txt")
        y_init_bbox_path = os.path.join(input_path, "init_rect.txt")
        if os.path.exists(y_true_path):
            self.y_true = load_rects(y_true_path)
            self.init_bbox = self.y_true[0]
        elif os.path.exists(y_init_bbox_path):
            self.init_bbox = load_rects(y_init_bbox_path)
            self.y_true = None
        else:
            assert False, "ground truth or inint rect required"

        # self.tags = load_tags(os.path.join(input_path, "description.txt")) not all scenes have a description
        self.name = scene_name
        self.hsi = ImageLoader(input_path, camera_type)
        self.falsecolor = ImageLoader(input_path_falsecolor, camera_type + "-FalseColor")
        self.shape = self.falsecolor[0].shape[:2]

    def __len__(self):
        return len(self.hsi)


class HOTDataset:

    def __init__(self, dataset_path, camera_type):
        self.dataset_path = dataset_path
        self.camera_type = camera_type
        self.dataset_type = Path(self.dataset_path).name # training or validation or ranking
        if camera_type not in CAMERA_TYPES:
            raise ValueError(f"Camera type not one of : {CAMERA_TYPES}")
        self.scenes = os.listdir(os.path.join(self.dataset_path, self.camera_type))

    def get_scene(self, name):
        return Scene(self.dataset_path, self.camera_type, name)

    def list_scenes(self):
        return self.scenes 

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.get_scene(self.scenes[i])


class HOTDatasetMultiCam:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_type = Path(self.dataset_path).name # training or validation
        
        self.datasets = []
        self.scenes = []
        for i,camera_type in enumerate(CAMERA_TYPES):
            d = HOTDataset(dataset_path, camera_type)
            self.datasets.append(d) 
            self.scenes += [ (i, name) for name in d.list_scenes() ]
            
    def __getitem__(self, i):
        d, name = self.scenes[i]
        return self.datasets[d].get_scene(name)

    def __len__(self):
        return len(self.scenes)

    def cam(self, camera_type):
        i = CAMERA_TYPES.index(camera_type)
        return self.datasets[i]


