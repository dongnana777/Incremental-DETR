# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import os
import cv2
import numpy as np
import random
from torchvision.transforms import transforms
from PIL import ImageFilter

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1,
                 cache_dir=None):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_samples = len(self.coco.getImgIds())
        self.strategy = 'topk'
        self.max_prop = 30
        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        self.files = []
        self.cache_dir = cache_dir
        for (troot, _, files) in os.walk(img_folder, followlinks=True):
            for f in files:
                if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
                    path = os.path.join(troot, f)
                    self.files.append(path)
                else:
                    continue
        print(f'num of files:{len(self.files)}')

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        w, h = img.size

        item = idx
        if self.strategy == 'topk':
            boxes = self.load_from_cache(item, img, h, w)
            boxes = boxes[:self.max_prop]
        elif self.strategy == 'mc':
            boxes = self.load_from_cache(item, img, h, w)
            boxes_indicators = np.where(np.random.binomial(1, p=self.dist2[:len(boxes)]))[0]
            boxes = boxes[boxes_indicators]
        elif self.strategy == "random":
            boxes = self.load_from_cache(random.choice(range(self.files)), None, None, None)  # relies on cache for now
            boxes = boxes[:self.max_prop]
        else:
            raise ValueError("No such strategy")

        # if len(boxes) < 2:
        #     return self.__getitem__(random.randint(0, len(self.files) - 1))

        target_select = {'orig_size': torch.as_tensor([int(h), int(w)]), 'size': torch.as_tensor([int(h), int(w)])}
        target_select['boxes'] = torch.tensor(boxes)
        target_select['iscrowd'] = torch.zeros(len(target_select['boxes']))
        target_select['area'] = target_select['boxes'][..., 2] * target_select['boxes'][..., 3]
        target_select['labels'] = torch.ones(len(target_select['boxes'])).long()*91

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target, target_select)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

            # if len(target['boxes']) < 2:
            #     return self.__getitem__(random.randint(0, len(self.files) - 1))

        return img, target

    def load_from_cache(self, item, img, h, w):
        fn = self.files[item].split('/')[-1].split('.')[0] + '.npy'
        fp = os.path.join(self.cache_dir, fn)
        try:
            with open(fp, 'rb') as f:
                boxes = np.load(f)
        except FileNotFoundError:
            boxes = selective_search(img, h, w)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, 'wb') as f:
                np.save(f, boxes)
        return boxes


def selective_search(img, h, w, res_size=None):
    img_det = np.array(img)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    if res_size is not None:
        img_det = cv2.resize(img_det, (res_size, res_size))

    ss.setBaseImage(img_det)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process().astype('long')

    if res_size is not None:
        boxes /= res_size
        boxes *= np.array([w, h, w, h])

    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, target_select):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        # print(classes)
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        classes = classes[keep]

        # if min(classes.shape) == 0:
        #     continue

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["iscrowd"] = iscrowd[keep]
        target["area"] = area[keep]
        target["num_gt"] = torch.tensor([target["boxes"].size(0)])
        target["boxes"] = torch.cat((target["boxes"], target_select["boxes"]), dim=0)
        target["labels"] = torch.cat((target["labels"], target_select["labels"]), dim=0)
        target["iscrowd"] = torch.cat((target["iscrowd"], target_select["iscrowd"]), dim=0)
        target["area"] = torch.cat((target["area"], target_select["area"]), dim=0)

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / "coco2017_inc" / 'base_train.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),

    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),
                            cache_dir=os.path.join(args.coco_path, 'cache'))


    return dataset
