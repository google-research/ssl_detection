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

#!/bin/bash
"""Generate labeled and unlabeled data for coco train.

Example:
python3 object_detection/prepare_coco_data.py
"""

import argparse
import numpy as np
import json
import os

assert 'COCODIR' in os.environ, 'Set COCODIR in bash'
DATA_DIR = os.environ['COCODIR']


def prepare_coco_data(seed=1, percent=10.0, version=2017):
  """Prepare COCO data for Semi-supervised learning

  Args:
    seed: random seed for data split
    percent: percentage of labeled data
    version: COCO data version
  """
  def _save_anno(name, images, annotations):
    """Save annotation
    """
    print('>> Processing data {}.json saved ({} images {} annotations)'.format(
        name, len(images), len(annotations)))
    new_anno = {}
    new_anno['images'] = images
    new_anno['annotations'] = annotations
    new_anno['licenses'] = anno['licenses']
    new_anno['categories'] = anno['categories']
    new_anno['info'] = anno['info']
    path = '{}/{}'.format(COCOANNODIR, 'semi_supervised')
    if not os.path.exists(path):
      os.mkdir(path)

    with open(
        '{root}/{folder}/{save_name}.json'.format(
            save_name=name, root=COCOANNODIR, folder='semi_supervised'),
        'w') as f:
      json.dump(new_anno, f)
    print('>> Data {}.json saved ({} images {} annotations)'.format(
        name, len(images), len(annotations)))

  np.random.seed(seed)
  COCOANNODIR = os.path.join(DATA_DIR, 'annotations')

  anno = json.load(open(os.path.join(COCOANNODIR,
            'instances_train{}.json'.format(version))))

  image_list = anno['images']
  labeled_tot = int(percent / 100. * len(image_list))
  labeled_ind = np.random.choice(range(len(image_list)), size=labeled_tot)
  labeled_id = []
  labeled_images = []
  unlabeled_images = []
  labeled_ind = set(labeled_ind)
  for i in range(len(image_list)):
    if i in labeled_ind:
      labeled_images.append(image_list[i])
      labeled_id.append(image_list[i]['id'])
    else:
      unlabeled_images.append(image_list[i])

  # get all annotations of labeled images
  labeled_id = set(labeled_id)
  labeled_annotations = []
  unlabeled_annotations = []
  for an in anno['annotations']:
    if an['image_id'] in labeled_id:
      labeled_annotations.append(an)
    else:
      unlabeled_annotations.append(an)

  # save labeled and unlabeled
  save_name = 'instances_train{version}.{seed}@{tot}'.format(
      version=version, seed=seed, tot=int(percent))
  _save_anno(save_name, labeled_images, labeled_annotations)
  save_name = 'instances_train{version}.{seed}@{tot}-unlabeled'.format(
      version=version, seed=seed, tot=int(percent))
  _save_anno(save_name, unlabeled_images, unlabeled_annotations)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--percent', type=float, default=10)
  parser.add_argument('--version', type=int, default=2017)
  parser.add_argument('--seed', type=int, help='seed', default=1)

  args = parser.parse_args()
  prepare_coco_data(args.seed, args.percent, args.version)
