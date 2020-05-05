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
"""Helper functions."""

from tensorpack.callbacks.base import Callback
from tensorpack.compat import tfv1 as tf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger


def add_moving_summary_no_nan(x, name=None):
  if not name:
    name = x.name
  add_moving_summary(
      tf.cond(
          tf.math.is_nan(x),
          lambda: tf.constant(0, tf.float32),
          lambda: x,
          name=name))


class WeightSyncCallBack(Callback):
  """Sync weight from source scope to target scope."""

  def __init__(self, schedule, src_scope='stg2'):
    self.schedule = schedule
    self.src_scope = src_scope
    self.assign_ops = []

  def _setup_graph(self):
    # get weight copy ops
    trainable_collection = tf.get_collection_ref(
        tf.GraphKeys.TRAINABLE_VARIABLES)
    mirrored_collection_name_map = dict()
    for var in trainable_collection:
      mirrored_collection_name_map[var.name.replace(self.src_scope + '/',
                                                    '')] = var
    mirrored_collection_name_set = set(mirrored_collection_name_map.keys())
    model_collection = tf.get_collection_ref(tf.GraphKeys.MODEL_VARIABLES)
    assign_ops = []
    for var in model_collection:
      if var.name in mirrored_collection_name_set:
        op = var.assign(mirrored_collection_name_map[var.name])
        assign_ops.append(op)
    self.assign_ops.extend(assign_ops)
    assert len(assign_ops) == len(trainable_collection)
    logger.info(
        '[WeightSyncCallBack] Create {} assign ops for WeightSyncCallBack, schedule = {}'
        .format(len(assign_ops), self.schedule))

  def _sync_weight(self):
    sess = tf.get_default_session()
    sess.run(self.assign_ops)
    logger.info('[WeightSyncCallBack] Sync weight at epoch {}'.format(
        self.epoch_num))

  def _before_epoch(self):
    if self.epoch_num in self.schedule:
      self._sync_weight()


class PathLog(Callback):
  """Path logging callback."""
  
  def __init__(self, path):
    self.path = path

  def before_train(self):
    self._after_epoch()

  def _after_epoch(self):
    logger.info('-' * 100)
    logger.info('Model save path: {}'.format(self.path))
    logger.info('-' * 100)
