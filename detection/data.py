# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: utf-8 -*-
# File: data.py

import copy
import itertools
import numpy as np
import cv2
from tabulate import tabulate
from termcolor import colored
import shutil
import os
import pdb
import deepdish as dd
from tqdm import tqdm
from tensorpack.dataflow import (
    DataFromList,
    MapData,
    MapDataComponent,
    MultiProcessMapData,
    MultiThreadMapData,
    TestDataSpeed,
    imgaug,
)
from tensorpack.utils import logger
from tensorpack.utils.argtools import log_once
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.utils import viz

from FasterRCNN.dataset import DatasetRegistry
from FasterRCNN.utils.np_box_ops import area as np_area
from FasterRCNN.data import TrainingDataPreprocessor, print_class_histogram, get_eval_dataflow
from FasterRCNN.common import (CustomResize, DataFromListOfDict, box_to_point8,
                               filter_boxes_inside_shape, np_iou, point8_to_box,
                               polygons_to_mask)

from config import finalize_configs
from config import config as cfg
from dataset import register_coco, register_voc
from utils.augmentation import RandomAugmentBBox
from utils.augmentation import bb_to_array, array_to_bb


def remove_empty_boxes(boxes):
  areas = np_area(boxes)
  mask = areas > 0
  return boxes[mask], mask


class DataFrom2List(RNGDataFlow):
  """ Wrap a list of datapoints to a DataFlow"""

  def __init__(self, lst, lst2, shuffle=True):
    """
        Args:
            lst (list): input list with labeled data. Each element is a
              datapoint.
            lst2 (list): input list with unlabled data. Each element is a
              datapoint.
            shuffle (bool): shuffle data.
    """
    super(RNGDataFlow, self).__init__()
    self.lst = lst
    self.lst2 = lst2
    #assert len(lst2) > len(lst), 'lst2 (for unlabeld data) should be bigger: {} < {}'.format(len(lst2),len(lst))
    self.shuffle = shuffle

  def __len__(self):
    # zizhaoz: we use unlabeled data as lengh of totaly dataset
    return len(self.lst2)

  def __iter__(self):
    if not self.shuffle:
      yield from [(a, b) for a, b in zip(itertools.cycle(self.lst), self.lst2)]
    else:
      idxs = np.arange(len(self.lst))
      idxs2 = np.arange(len(self.lst2))
      self.rng.shuffle(idxs)
      self.rng.shuffle(idxs2)
      for k, k2 in zip(itertools.cycle(idxs), idxs2):
        yield (self.lst[k], self.lst2[k2])


class TrainingDataPreprocessorAug(TrainingDataPreprocessor):
  """Generalized from TrainingDataPreprocessor with strong augmentation."""

  def __init__(self, cfg):
    self.cfg = cfg
    self.aug_weak = imgaug.AugmentorList([
        CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
        imgaug.Flip(horiz=True)
    ])
    self.aug_type = cfg.TRAIN.AUGTYPE_LAB
    self.aug_strong = RandomAugmentBBox(aug_type=cfg.TRAIN.AUGTYPE_LAB)
    logger.info("Use affine-enabled TrainingDataPreprocessor_aug")

  def __call__(self, roidb):  #
    fname, boxes, klass, is_crowd = roidb["file_name"], roidb["boxes"], roidb[
        "class"], roidb["is_crowd"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4, boxes.shape
    boxes = np.copy(boxes)
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    assert im is not None, fname
    im = im.astype("float32")
    height, width = im.shape[:2]
    # assume floatbox as input
    assert boxes.dtype == np.float32, "Loader has to return float32 boxes!"

    if not self.cfg.DATA.ABSOLUTE_COORD:
      boxes[:, 0::2] *= width
      boxes[:, 1::2] *= height

    ret = {}

    tfms = self.aug_weak.get_transform(im)
    im = tfms.apply_image(im)
    points = box_to_point8(boxes)
    points = tfms.apply_coords(points)
    boxes = point8_to_box(points)
    h, w = im.shape[:2]
    if self.aug_type != "default":
      boxes_backup = boxes.copy()
      try:
        assert len(boxes) > 0, "boxes after resizing becomes to zero"
        assert np.sum(np_area(boxes)) > 0, "boxes are all zero area!"
        bbs = array_to_bb(boxes)
        images_aug, bbs_aug, _ = self.aug_strong(
            images=[im], bounding_boxes=[bbs], n_real_box=len(bbs))

        # convert to gt boxes array
        boxes = bb_to_array(bbs_aug[0])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

        # after affine, some boxes can be zero area. Let's remove them and their corresponding info
        boxes, mask = remove_empty_boxes(boxes)
        klass = klass[mask]
        assert len(
            klass) > 0, "Empty boxes and kclass after removing empty ones"
        is_crowd = np.array([0] * len(klass))  # do not ahve crowd annotations
        assert klass.max() <= self.cfg.DATA.NUM_CATEGORY, \
            "Invalid category {}!".format(klass.max())
        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"
        im = images_aug[0]
      except Exception as e:
        logger.warn("Error catched " + str(e) + "\n Use non-augmented data.")
        boxes = boxes_backup

    ret["image"] = im

    try:
      # Add rpn data to dataflow:
      if self.cfg.MODE_FPN:
        multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(
            im, boxes, is_crowd)
        for i, (anchor_labels,
                anchor_boxes) in enumerate(multilevel_anchor_inputs):
          ret["anchor_labels_lvl{}".format(i + 2)] = anchor_labels
          ret["anchor_boxes_lvl{}".format(i + 2)] = anchor_boxes
      else:
        ret["anchor_labels"], ret["anchor_boxes"] = self.get_rpn_anchor_input(
            im, boxes, is_crowd)

      boxes = boxes[is_crowd == 0]  # skip crowd boxes in training target
      klass = klass[is_crowd == 0]
      ret["gt_boxes"] = boxes
      ret["gt_labels"] = klass

    except Exception as e:
      log_once("Input {} is filtered for training: {}".format(fname, str(e)),
               "warn")
      return None

    return ret


class TrainingDataPreprocessorSSlAug(TrainingDataPreprocessor):
  """Generalized from TrainingDataPreprocessor.

    It supports loading paired (labeled, unlabeled) data flow (e.g.
    DataFrom2List). It loads pre-generated pseudo labels from disk.
    """

  def __init__(self, cfg, confidence, pseudo_targets):
    self.cfg = cfg
    self.aug = imgaug.AugmentorList([
        CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
        imgaug.Flip(horiz=True)
    ])

    self.resize = imgaug.AugmentorList([
        CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
    ])

    self.aug_strong = RandomAugmentBBox(aug_type=cfg.TRAIN.AUGTYPE)
    self.aug_strong_labeled = RandomAugmentBBox(aug_type=cfg.TRAIN.AUGTYPE_LAB)
    self.labeled_augment_type = cfg.TRAIN.AUGTYPE_LAB
    self.unlabeled_augment_type = cfg.TRAIN.AUGTYPE

    self.confidence = confidence
    logger.info(
        "Use TrainingDataPreprocessor6 (using offline generated pseudo labels)")
    self.pseudo_targets = pseudo_targets

  def get_pseudo_gt(self, img_id):
    """
        {'proposals_boxes':  (?,4)
        'proposals_scores':  (?,)
        'boxes': (?,4)
        'scores': (?,)
        'labels': (?,)
        }
        """
    if img_id not in self.pseudo_targets:
      return None
    pseudo_gt = self.pseudo_targets[img_id]
    true_confidence = np.minimum(self.confidence, pseudo_gt["scores"].max())
    mask = pseudo_gt["scores"] >= true_confidence
    masked_target = {}
    for k, v in pseudo_gt.items():
      if len(v) == len(mask):
        # move to float32 in order to prevent area is equal to inf
        masked_target[k] = v[mask].astype(np.float32)
      else:
        masked_target[k] = v.astype(np.float32)
    return masked_target

  def __call__(self, roidbs):  #
    # roidbs2 repsect to unlabeled data

    def prepare_data(roidb, aug, aug_type="default", is_unlabled=False):
      fname, boxes, klass, is_crowd, img_id = roidb["file_name"], roidb[
          "boxes"], roidb["class"], roidb["is_crowd"], roidb["image_id"]
      assert boxes.ndim == 2 and boxes.shape[1] == 4, boxes.shape
      boxes = np.copy(boxes)
      im = cv2.imread(fname, cv2.IMREAD_COLOR)
      assert im is not None, fname
      im = im.astype("float32")
      height, width = im.shape[:2]
      # assume floatbox as input
      assert boxes.dtype == np.float32, "Loader has to return float32 boxes!"

      if not self.cfg.DATA.ABSOLUTE_COORD:
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height

      ret = {}
      if not is_unlabled and aug_type == "default":
        tfms = aug.get_transform(im)
        im = tfms.apply_image(im)
        points = box_to_point8(boxes)
        points = tfms.apply_coords(points)
        boxes = point8_to_box(points)
      else:
        # It is strong augmentation
        # Load box informaiton from disk
        if is_unlabled:
          pseudo_target = self.get_pseudo_gt(img_id)
          # has no pseudo target found
          assert pseudo_target is not None
          boxes = pseudo_target["boxes"]
          klass = pseudo_target["labels"].astype(np.int32)
          assert len(boxes) > 0, "boxes after thresholding becomes to zero"
          is_crowd = np.array([0] * len(klass))  # do not ahve crowd annotations
        else:
          # it is labeled data, use boxes loaded from roidb, klass, is_crowd
          pass

        if aug_type == "default":
          # use default augmentations, only happend for unlabeled data
          tfms = self.aug.get_transform(im)
          im = tfms.apply_image(im)
          points = box_to_point8(boxes)
          points = tfms.apply_coords(points)
          boxes = point8_to_box(points)
          # is_crowd = np.array([0]*len(klass)) # do not ahve crowd annotations
        else:
          # use strong augmentation with extra packages
          # resize first
          tfms = self.resize.get_transform(im)
          im = tfms.apply_image(im)
          points = box_to_point8(boxes)
          points = tfms.apply_coords(points)
          boxes = point8_to_box(points)
          boxes_backup = boxes.copy()
          h, w = im.shape[:2]

          # strong augmentation
          try:
            assert len(boxes) > 0, "boxes after resizing becomes to zero"
            assert np.sum(np_area(boxes)) > 0, "boxes are all zero area!"
            bbs = array_to_bb(boxes)
            images_aug, bbs_aug, _ = aug(
                images=[im], bounding_boxes=[bbs], n_real_box=len(bbs))

            # # convert to gt boxes array
            boxes = bb_to_array(bbs_aug[0])

            boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, h)

            # after affine, some boxes can be zero area. Let's remove them and their corresponding info
            boxes, mask = remove_empty_boxes(boxes)
            klass = klass[mask]
            is_crowd = is_crowd[mask]
            assert len(
                klass) > 0, "Empty boxes and kclass after removing empty ones"
            assert klass.max() <= self.cfg.DATA.NUM_CATEGORY, \
                "Invalid category {}!".format(klass.max())
            assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"
            im = images_aug[0]
          except Exception as e:
            # if augmentation makes the boxes become empty, we switch to
            # non-augmented one
            # logger.warn("Error catched " + str(e) +
            #             "\n Use non-augmented data.")
            boxes = boxes_backup

      ret["image"] = im

      # Add rpn data to dataflow:
      if self.cfg.MODE_FPN:
        multilevel_anchor_inputs = self.get_multilevel_rpn_anchor_input(
            im, boxes, is_crowd)
        for i, (anchor_labels,
                anchor_boxes) in enumerate(multilevel_anchor_inputs):
          ret["anchor_labels_lvl{}".format(i + 2)] = anchor_labels
          ret["anchor_boxes_lvl{}".format(i + 2)] = anchor_boxes
      else:
        ret["anchor_labels"], ret["anchor_boxes"] = self.get_rpn_anchor_input(
            im, boxes, is_crowd)

      boxes = boxes[is_crowd == 0]  # skip crowd boxes in training target
      klass = klass[is_crowd == 0]
      ret["gt_boxes"] = boxes
      ret["gt_labels"] = klass

      if is_unlabled:
        ret["proposals_boxes"] = pseudo_target["proposals_boxes"]
        # ret["proposals_scores"] = pseudo_target['proposals_scores']
      return ret

    try:
      roidb, roidb_u = roidbs
      results = {}
      if self.labeled_augment_type == "default":
        results.update(prepare_data(roidb, self.aug, is_unlabled=False))
      else:
        results.update(
            prepare_data(
                roidb,
                self.aug_strong_labeled,
                aug_type=self.labeled_augment_type,
                is_unlabled=False))
      # strong augmentation
      res_u = {}
      for k, v in prepare_data(
          roidb_u,
          self.aug_strong,
          aug_type=self.unlabeled_augment_type,
          is_unlabled=True).items():
        res_u[k + "_strong"] = v

      results.update(res_u)
    except Exception as e:
      logger.warn("Input is filtered " + str(e))
      return None

    return results


def get_train_dataflow():
  """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)

    If MODE_MASK, gt_masks: (N, h, w)
    """

  roidbs = list(
      itertools.chain.from_iterable(
          DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
  print_class_histogram(roidbs)

  # Filter out images that have no gt boxes, but this filter shall not be applied for testing.
  # The model does support training with empty images, but it is not useful for COCO.
  num = len(roidbs)
  roidbs = list(
      filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, roidbs))
  logger.info(
      "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}"
      .format(num - len(roidbs), len(roidbs)))

  ds = DataFromList(roidbs, shuffle=True)
  preprocess = TrainingDataPreprocessorAug(cfg)

  if cfg.DATA.NUM_WORKERS > 0:
    if cfg.TRAINER == "horovod":
      buffer_size = cfg.DATA.NUM_WORKERS * 10  # one dataflow for each process, therefore don't need large buffer
      ds = MultiThreadMapData(
          ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
      # MPI does not like fork()
    else:
      buffer_size = cfg.DATA.NUM_WORKERS * 20
      ds = MultiProcessMapData(
          ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
  else:
    ds = MapData(ds, preprocess)
  return ds


def get_eval_unlabeled_dataflow(name, shard=0, num_shards=1, return_size=False):
  """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)

    If MODE_MASK, gt_masks: (N, h, w)
    """
  if isinstance(name, (list, tuple)) and len(name) > 1:
    if "VOC" not in name[0]:
      assert "VOC" not in name[
          1], "VOC has to be put before coco in cfg.DATA.TRAIN"
    roidbs = []
    for x in name:
      _roidbs = DatasetRegistry.get(x).training_roidbs()
      print_class_histogram(_roidbs)
      roidbs.extend(_roidbs)
    # roidbs = list(itertools.chain.from_iterable(DatasetRegistry.get(x).training_roidbs() for x in name))
    logger.info("Merged roidbs from {}".format(name))
    print_class_histogram(roidbs)
  else:
    if isinstance(name, (list, tuple)):
      name = name[0]
    roidbs = DatasetRegistry.get(name).training_roidbs()
    print_class_histogram(roidbs)

  num_imgs = len(roidbs)
  img_per_shard = num_imgs // num_shards
  img_range = (shard * img_per_shard, (shard + 1) *
               img_per_shard if shard + 1 < num_shards else num_imgs)
  logger.info("Found {} images for inference.".format(img_range[1] -
                                                      img_range[0] + 1))

  # no filter for training
  ds = DataFromListOfDict(roidbs[img_range[0]:img_range[1]],
                          ["file_name", "image_id"])

  def f(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    assert im is not None, fname
    return im

  ds = MapDataComponent(ds, f, 0)
  # Evaluation itself may be multi-threaded, therefore don't add prefetch
  # here.

  if return_size:
    return ds, num_imgs
  return ds


def get_train_dataflow_w_unlabeled(load_path):
  """
    Return a training dataflow. Each datapoint consists of the following:

    An image: (h, w, 3),

    1 or more pairs of (anchor_labels, anchor_boxes):
    anchor_labels: (h', w', NA)
    anchor_boxes: (h', w', NA, 4)

    gt_boxes: (N, 4)
    gt_labels: (N,)

    If MODE_MASK, gt_masks: (N, h, w)
    """
  assert os.path.isfile(load_path), "{} does not find".format(load_path)
  roidbs = list(
      itertools.chain.from_iterable(
          DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
  print_class_histogram(roidbs)

  if "VOC" in cfg.DATA.TRAIN[0]:
    roidbs_u = list(
        itertools.chain.from_iterable(
            DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.UNLABEL))
    unlabled2017_used = False
  else:
    unlabled2017_used = np.any(["@" not in x for x in cfg.DATA.TRAIN])

    def prase_name(x):
      if not unlabled2017_used:
        assert "@" in load_path, ("{}: Did you use wrong pseudo_data.py for "
                                  "this model?").format(load_path)
        return x + "-unlabeled"
      else:
        # return coco2017 unlabeled data
        return "coco_unlabeled2017"

    roidbs_u = list(
        itertools.chain.from_iterable(
            DatasetRegistry.get(prase_name(x)).training_roidbs()
            for x in cfg.DATA.TRAIN))
  print_class_histogram(roidbs_u)

  # Filter out images that have no gt boxes, but this filter shall not be applied for testing.
  # The model does support training with empty images, but it is not useful for COCO.
  def remove_no_box_data(_roidbs, filter_fn, dset):
    num = len(_roidbs)
    _roidbs = filter_fn(_roidbs)
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total {} #images for training: {}"
        .format(num - len(_roidbs), dset, len(_roidbs)))
    return _roidbs

  roidbs = remove_no_box_data(
      roidbs, lambda x: list(
          filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, x)),
      "labeled")
  # load unlabeled
  if unlabled2017_used:
    assert "@" not in load_path, "Did you use the wrong pseudo path"
  pseudo_targets = dd.io.load(load_path)
  logger.info("Loaded {} pseudo targets from {}".format(
      len(pseudo_targets), load_path))
  roidbs_u = remove_no_box_data(
      roidbs_u, lambda x: list(
          filter(lambda img: len(pseudo_targets[img["image_id"]]["boxes"]) > 0,
                 x)), "unlabeled")
  preprocess = TrainingDataPreprocessorSSlAug(
      cfg, confidence=cfg.TRAIN.CONFIDENCE, pseudo_targets=pseudo_targets)

  ds = DataFrom2List(roidbs, roidbs_u, shuffle=True)

  if cfg.DATA.NUM_WORKERS > 0:
    if cfg.TRAINER == "horovod":
      buffer_size = cfg.DATA.NUM_WORKERS * 10
      ds = MultiThreadMapData(
          ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
    else:
      buffer_size = cfg.DATA.NUM_WORKERS * 20
      ds = MultiProcessMapData(
          ds, cfg.DATA.NUM_WORKERS, preprocess, buffer_size=buffer_size)
  else:
    ds = MapData(ds, preprocess)
  return ds


def visualize_dataflow2(cfg, unlabled2017_used=True, VISPATH="./", maxvis=50):
  """Visualize the dataflow with labeled and unlabled strong augmentation."""

  def prase_name(x):
    if not unlabled2017_used:
      return x + "-unlabeled"
    else:  # return coco2017 unlabeled data
      return "coco_unlabeled2017"

  def remove_no_box_data(_roidbs, filter_fn):
    num = len(_roidbs)
    _roidbs = filter_fn(_roidbs)
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}"
        .format(num - len(_roidbs), len(_roidbs)))
    return _roidbs

  pseudo_path = os.path.join(os.environ["PSEUDO_PATH"], "pseudo_data.npy")
  pseudo_targets = dd.io.load(pseudo_path)

  roidbs = list(
      itertools.chain.from_iterable(
          DatasetRegistry.get(x).training_roidbs() for x in cfg.DATA.TRAIN))
  roidbs_u = list(
      itertools.chain.from_iterable(
          DatasetRegistry.get(prase_name(x)).training_roidbs()
          for x in cfg.DATA.TRAIN))
  roidbs = remove_no_box_data(
      roidbs, lambda x: list(
          filter(lambda img: len(img["boxes"][img["is_crowd"] == 0]) > 0, x)))
  roidbs_u = remove_no_box_data(
      roidbs_u, lambda x: list(
          filter(lambda img: len(pseudo_targets[img["image_id"]]["boxes"]) > 0,
                 x)))

  print_class_histogram(roidbs)
  print_class_histogram(roidbs_u)

  preprocess = TrainingDataPreprocessorSSlAug(
      cfg, confidence=cfg.TRAIN.CONFIDENCE, pseudo_targets=pseudo_targets)
  for jj, (rob, robu) in tqdm(enumerate(zip(roidbs, roidbs_u))):
    data = preprocess((rob, robu))
    # import pdb; pdb.set_trace()
    nn = len(pseudo_targets[robu["image_id"]]["boxes"])
    if data is None or len(data["gt_boxes_strong"]) == 0:
      print("empty annotation, {} (original {})".format(jj, nn))
      continue

    ims = viz.draw_boxes(data["image"], data["gt_boxes"],
                         [str(a) for a in data["gt_labels"]])

    ims_t = viz.draw_boxes(data["image_strong"], data["gt_boxes_strong"], [
        str(a) for a in data["gt_labels_strong"][:len(data["gt_boxes_strong"])]
    ])
    ims = cv2.resize(ims, (ims_t.shape[1], ims_t.shape[0]))
    vis = np.concatenate((ims, ims_t), axis=1)
    if not os.path.exists(
        os.path.dirname(os.path.join(VISPATH, "result_{}.jpeg".format(jj)))):
      os.makedirs(
          os.path.dirname(os.path.join(VISPATH, "result_{}.jpeg".format(jj))))
    assert cv2.imwrite(os.path.join(VISPATH, "result_{}.jpeg".format(jj)), vis)

    if jj > maxvis:
      break


if __name__ == "__main__":
  # visualize augmented data
  # Follow README.md to set necessary environment variables, then
  # CUDA_VISIBLE_DEVICES=0 VISPATH=<your-save-path> AUGTYPE='strong' python data.py
  cfg.DATA.NUM_WORKERS = 0
  register_coco(os.path.expanduser(os.environ["DATADIR"]))
  finalize_configs(True)

  cfg.DATA.TRAIN = ("coco_unlabeled2017",)
  cfg.TRAIN.AUGTYPE = os.environ["AUGTYPE"]
  VISPATH = os.environ["VISPATH"]
  VISPATH = os.path.join(os.environ["VISPATH"], str(cfg.TRAIN.AUGTYPE))
  if os.path.isdir(VISPATH):
    shutil.rmtree(VISPATH)
  os.makedirs(VISPATH)
  cfg.TRAIN.CONFIDENCE = 0.5
  visualize_dataflow2(cfg, VISPATH)
