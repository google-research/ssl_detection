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
"""Object detection strong augmentation utilities."""
# pylint: disable=g-explicit-length-test

from auto_augment import augmentations
from FasterRCNN.utils import np_box_ops

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters.geometric import Affine
import numpy as np

from tensorpack.utils import logger


RANDOM_COLOR_POLICY_OPS = (
    'Identity',
    'AutoContrast',
    'Equalize',
    'Solarize',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'Posterize',
)

# for image size == 800, 0.1 is 80.
CUTOUT = iaa.Cutout(nb_iterations=(1, 5), size=[0, 0.2], squared=True)

DEGREE = 30
AFFINE_TRANSFORM = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(  # TranslateX
                translate_percent={'x': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # TranslateY
                translate_percent={'y': (-0.1, 0.1)},
                order=[0, 1],
                cval=125,
            ),
            Affine(  # Rotate
                rotate=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
            Affine(  # ShearX and ShareY
                shear=(-DEGREE, DEGREE),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

AFFINE_TRANSFORM_WEAK = iaa.Sequential(
    [
        iaa.OneOf([
            Affine(
                translate_percent={'x': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                translate_percent={'y': (-0.05, 0.05)},
                order=[0, 1],
                cval=125,
            ),
            Affine(
                rotate=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
            Affine(
                shear=(-10, 10),
                order=[0, 1],
                cval=125,
            ),
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True)

COLOR = iaa.Sequential(
    [
        iaa.OneOf(  # apply one color transformation
            [
                iaa.Add((0, 0)),  # identity transform
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),
                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                # Improve or worsen the contrast of images.
                iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),
            ])
    ],
    random_order=True)


def bb_to_array(bbs):
  coords = []
  for bb in bbs:
    coords.append([bb.x1, bb.y1, bb.x2, bb.y2])
  coords = np.array(coords)

  return coords


def array_to_bb(coords):
  # convert to bbs
  bbs = []
  for b in coords:
    bbs.append(ia.BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
  return bbs


class RandomAugmentBBox(object):
  """Augmentation class."""

  def __init__(self,
               aug_type='strong',
               magnitude=10,
               weighted_inbox_selection=False):
    self.affine_aug_op = AFFINE_TRANSFORM
    # for inbox affine, we use small degree
    self.inbox_affine_aug_op = AFFINE_TRANSFORM_WEAK
    self.jitter_aug_op = COLOR
    self.cutout_op = CUTOUT
    self.magnitude = magnitude
    self.aug_type = aug_type
    self.weighted_inbox_selection = weighted_inbox_selection

    # self.augment_fn is a list of list (i.g. [[],...]), each item is a list of
    # augmentation will be randomly picked up. Th total items determine the
    # number of layers of augmentation to apply
    # Note: put cutout_augment at last if used.
    if aug_type == 'strong':
      # followd by cutout
      self.augment_fn = [[self.color_augment],
                         [self.bbox_affine_transform, self.affine_transform],
                         [self.cutout_augment]]
    elif aug_type == 'default':
      self.augment_fn = []
    else:
      raise NotImplementedError('aug_type {} does not exist'.format(aug_type))

    logger.info('-' * 100)
    logger.info('Augmentation type {}: {}'.format(aug_type, self.augment_fn))
    logger.info('-' * 100)

  def normaize(self, x):
    x /= 255.0
    x = x / 0.5 - 1.0
    return x

  def unnormaize(self, x):
    x = (x + 1.0) * 0.5 * 255.0
    return x

  def numpy_apply_policies(self, arglist):
    x, policies = arglist
    re = []
    for y, policy in zip(x, policies):
      # apply_policy has input to have images [-1, 1]
      y_a = augmentations.apply_policy(policy, self.normaize(y))
      y_a = np.reshape(y_a, y.shape)
      y_a = self.unnormaize(y_a)
      re.append(y_a)
    return np.stack(re).astype('f')

  def bbox_affine_transform(self, images, bounding_boxes, **kwargs):
    """In-box affine transformation."""
    real_box_n = kwargs['n_real_box']
    shape = images[0].shape
    for im, boxes in zip(images, bounding_boxes):
      boxes = bb_to_array(boxes)
      # large area has better probability to be sampled
      if self.weighted_inbox_selection:
        area = np_box_ops.area(boxes[:real_box_n])
        k = np.random.choice([i for i in range(real_box_n)],
                             1,
                             p=area / area.sum())[0]
      else:
        k = np.random.choice([i for i in range(real_box_n)], 1)[0]

      if len(boxes) > 0:
        box = boxes[k]
        im_crop = im[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
        im_paste = self.inbox_affine_aug_op(images=[im_crop])[0]
        # in-memory operation
        im[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = im_paste
    assert shape == images[0].shape
    return images, bounding_boxes, None

  def affine_transform(self, images, bounding_boxes, **kwargs):
    """Global affine transformation."""
    del kwargs
    shape = images[0].shape
    images_aug, bbs_aug = self.affine_aug_op(
        images=images, bounding_boxes=bounding_boxes)
    assert shape == images_aug[0].shape
    return images_aug, bbs_aug, None

  def jitter_augment(self, images, bounding_boxes=None, **kwargs):
    """Color jitters."""
    del kwargs
    images_aug = self.jitter_aug_op(images=images)
    return images_aug, bounding_boxes, None

  def cutout_augment(self, images, bounding_boxes=None, **kwargs):
    """Cutout augmentation."""
    del kwargs
    images_aug = self.cutout_op(images=images)
    return images_aug, bounding_boxes, None

  def color_augment(self, images, bounding_boxes=None, p=1.0, **kwargs):
    """RandAug color augmentation."""
    del kwargs
    policy = lambda: [(op, p, np.random.randint(1, self.magnitude))  # pylint: disable=g-long-lambda
                      for op in np.random.choice(RANDOM_COLOR_POLICY_OPS, 1)]
    images_aug = []
    shape = images[0].shape
    for x in range(len(images)):
      images_aug.append(
          self.numpy_apply_policies((images[x:x + 1], [policy()]))[0])
    assert shape == images_aug[0].shape
    if bounding_boxes is None:
      return images_aug
    return images_aug, bounding_boxes, None

  def __call__(self, images, bounding_boxes, **kwargs):

    # random order
    if len(self.augment_fn
          ) > 0 and self.augment_fn[-1][0].__name__ == 'cutout_augment':
      # put cutout in the last always
      naug = len(self.augment_fn)
      order = np.random.permutation(np.arange(naug - 1))
      order = np.concatenate([order, [naug - 1]], 0)
    else:
      order = np.random.permutation(np.arange(len(self.augment_fn)))

    # pylint: disable=invalid-name
    T = None
    for i in order:
      fns = self.augment_fn[i]
      fn = fns[np.random.randint(0, len(fns))]
      images, bounding_boxes, _T = fn(
          images=images, bounding_boxes=bounding_boxes, **kwargs)
      if _T is not None:
        T = _T
    return images, bounding_boxes, T
