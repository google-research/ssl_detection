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
# File: config.py

import numpy as np
import os
import pprint
import six
import re

from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu
from FasterRCNN.config import *
from FasterRCNN.config import config

_C = config
# The proposed method related hyperparams.
# The majority of config argumetns are defined at third_party/FasterRCNN/FasterRCNN/config.py

_C.TRAIN.STAGE = 1  # Stage of the training
_C.TRAIN.CONFIDENCE = 0.9  # Confidence threshold
_C.TRAIN.WU = 2.  # loss weight of unlabled data
_C.TRAIN.NO_PRN_LOSS = False  # disable RPN loss
_C.EVAL.PSEUDO_INFERENCE = False  # Doing pseduo labeling inference process, this makes inferense on orignal image size.
_C.TRAIN.AUGTYPE = 'strong'  # augmentation type for unlabeled data
_C.TRAIN.AUGTYPE_LAB = 'default'  # augmentation type for labeled data
_C.DATA.UNLABEL = ('',)  # extra unlabeled json files (not always used)

_C.freeze()  # avoid typo / wrong config keys
