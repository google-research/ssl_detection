# Copyright 2020 Google LLC

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

"""COCO dataset register."""
# pylint: disable=invalid-name
# pylint: disable=g-explicit-length-test
# pylint: redefined-outer-name
import json
import numpy as np
import os
import tqdm

from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from FasterRCNN.dataset import DatasetRegistry
from FasterRCNN.dataset import DatasetSplit
from FasterRCNN.dataset.coco import COCODetection as COCODetectionBase
from FasterRCNN.dataset.coco import register_coco as register_coco_supervised

__all__ = ["register_coco"]


# register semi-supervised splits for coco
SEMI_SUPERVISED_SPLITS = []
for seed in range(1, 6):
  for percent in [1, 2, 5, 10, 20, 30, 40, 50]:
    SEMI_SUPERVISED_SPLITS.append("train2017.{}@{}".format(seed, percent))
    # adding corresponding unlabeled one
    SEMI_SUPERVISED_SPLITS.append("train2017.{}@{}-unlabeled".format(
        seed, percent))
# 100% , unlab is with lab
SEMI_SUPERVISED_SPLITS.append("train2017.{}@{}-extra".format(0, 100))
SEMI_SUPERVISED_SPLITS.append("train2017.{}@{}-extra-unlabeled".format(0, 100))
# coco unlabled data
SEMI_SUPERVISED_SPLITS.append("unlabeled2017")
# coco 20 class unlabeled for voc
NUM_20CLASS = 1
SEMI_SUPERVISED_SPLITS.append("unlabeledtrainval20class")


class COCODetection(COCODetectionBase):
  """COCO class object.

  Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own coco-format, dataset, change this to an **empty dict**.
  """
  # handle a few special splits whose names do not match the directory names
  _INSTANCE_TO_BASEDIR = {
      "valminusminival2014": "val2014",
      "minival2014": "val2014",
      "val2017_100": "val2017",
      "unlabeled2017": "unlabeled2017",
      "train2017.{}@{}-extra".format(0, 100): "",
      "train2017.{}@{}-extra-unlabeled".format(0, 100): "",
  }

  def __init__(self, basedir, split):
    """Init.

    Args:
        basedir (str): root of the dataset which contains the subdirectories
          for each split and annotations
        split (str): the name of the split, e.g. "train2017". The split has
          to match an annotation file in "annotations/" and a directory of
          images.
    Examples:
        For a directory of this structure:  DIR/ annotations/
          instances_XX.json instances_YY.json XX/ YY/  use
          `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
    """
    for sp in SEMI_SUPERVISED_SPLITS:
      if sp not in self._INSTANCE_TO_BASEDIR:
        self._INSTANCE_TO_BASEDIR.update({str(sp): "train2017"})

    basedir = os.path.expanduser(basedir)
    self._imgdir = os.path.realpath(
        os.path.join(basedir, self._INSTANCE_TO_BASEDIR.get(split, split)))
    assert os.path.isdir(self._imgdir), "{} is not a directory!".format(
        self._imgdir)
    if split in SEMI_SUPERVISED_SPLITS:
      annotation_file = os.path.join(
          basedir,
          "annotations/semi_supervised/instances_{}.json".format(split))
    else:
      annotation_file = os.path.join(
          basedir, "annotations/instances_{}.json".format(split))
    assert os.path.isfile(annotation_file), annotation_file

    self.coco = COCO(annotation_file)
    self.annotation_file = annotation_file
    logger.info("Instances loaded from {}.".format(annotation_file))

  def eval_inference_results2(self,
                              results,
                              output=None,
                              threshold=None,
                              metric_only=False):
    # Compared with eval_inference_results, v2 version has an threshold
    # used to filter scores below. It is designed for SSL experiments.
    if not metric_only:
      if threshold is not None:
        logger.warn(
            "Use thresholding {} to filter final resulting boxes".format(
                threshold))
      continuous_id_to_COCO_id = {
          v: k for k, v in self.COCO_id_to_category_id.items()
      }
      n = 0
      final_results = []
      for res in results:
        # convert to COCO's incontinuous category id
        if res["category_id"] in continuous_id_to_COCO_id:
          res["category_id"] = continuous_id_to_COCO_id[res["category_id"]]

        if threshold is not None:
          if res["score"] < threshold:
            n += 1
            continue
        # COCO expects results in xywh format
        box = res["bbox"]
        box[2] -= box[0]
        box[3] -= box[1]
        res["bbox"] = [round(float(x), 3) for x in box]
        final_results.append(res)

      results = final_results
      if output is not None:
        if not os.path.exists(os.path.dirname(output)):
          os.makedirs(os.path.dirname(output))
        with open(output, "w") as f:
          json.dump(results, f)
        if threshold is not None:
          with open(output + "_boxcount.json", "w") as f:
            r = {"passed": len(results), "removed": n}
            print("Box thresholding stats: \n\t", r)
            json.dump(r, f)

    if len(results):
      metrics = self.print_coco_metrics(results)
      # save precision_recall data:
      precision_recall = self.cocoEval.precision_recall
      pr_path = os.path.join(os.path.split(output)[0], "precision_recall.npy")
      print("Saving precision_recall curve to {}".format(pr_path))
      np.save(pr_path, {"pr": precision_recall})
      # sometimes may crash if the results are empty?
      return metrics
    else:
      return {}


def register_coco(basedir):
  """Register COCO.

  Add COCO datasets like "coco_train201x" to the registry,
  so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.

  Note that train2017==trainval35k==train2014+val2014-minival2014, and
  val2017==minival2014.

  Args:
    basedir: root dir that saves datasets.
  """

  # 80 names for COCO
  # For your own coco-format dataset, change this.
  class_names = [
      "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
      "truck", "boat", "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
      "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
      "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
      "baseball bat", "baseball glove", "skateboard", "surfboard",
      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
      "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
      "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
      "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
      "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
      "hair drier", "toothbrush"
  ]  # noqa
  class_names = ["BG"] + class_names
  register_coco_supervised(basedir)

  for split in SEMI_SUPERVISED_SPLITS[:-NUM_20CLASS]:
    name = "coco_" + split
    DatasetRegistry.register(name, lambda x=split: COCODetection(basedir, x))
    DatasetRegistry.register_metadata(name, "class_names", class_names)

  logger.info("Register dataset {}".format(
      [a for a in DatasetRegistry._registry.keys()]))  # pylint: disable=protected-access

  assert os.environ["COCODIR"], "COCODIR environ variable is not set".format(
      os.environ["COCODIR"])
  # also register coco train set 20 class for voc experiments
  register_coco_for_voc(os.environ["COCODIR"])


class COCODetectionForVOC(COCODetection):
  """COCODetection for VOC."""
  # set to empty since this instances_unlabeledtrainval20class.json file has file_name with relative path to train2017 or val2017
  _INSTANCE_TO_BASEDIR = {"unlabeledtrainval20class": ""}
  # this mapping is obtained by running dataset/cls_mapping_coco_voc.py
  COCO_id_to_category_id = {
      64: 14,
      1: 3,
      2: 6,
      3: 10,
      4: 1,
      5: 16,
      6: 18,
      7: 9,
      72: 20,
      9: 8,
      67: 19,
      44: 17,
      16: 11,
      17: 12,
      18: 2,
      19: 4,
      20: 15,
      21: 7,
      62: 13,
      63: 5
  }


def register_coco_for_voc(basedir):
  class_names = [
      "person", "chair", "aeroplane", "bus", "cow", "bird", "motorbike", "boat",
      "car", "horse", "sofa", "pottedplant", "tvmonitor", "cat", "train",
      "bottle", "diningtable", "dog", "bicycle", "sheep"
  ]
  class_names = ["BG"] + class_names
  for split in SEMI_SUPERVISED_SPLITS[-NUM_20CLASS:]:
    name = "coco_" + split
    DatasetRegistry.register(
        name, lambda x=split: COCODetectionForVOC(basedir, x))
    DatasetRegistry.register_metadata(name, "class_names", class_names)

  logger.info("Register dataset {}".format(
      [a for a in DatasetRegistry._registry.keys()]))


if __name__ == "__main__":
  basedir = "<add-data-path>"
  c = COCODetection(basedir, "train2017")
  roidb = c.load(add_gt=True, add_mask=True)
  print("#Images:", len(roidb))
