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
"""Stage 2 training entrance."""
# pylint: disable=g-multiple-import
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
# pylint: disable=unused-variable
import argparse
import os
import warnings
import multiprocessing as mp
warnings.simplefilter('once', DeprecationWarning)

from tensorpack import *
from tensorpack.tfutils import collect_env_info
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.callbacks import JSONWriter
import tensorflow as tf
import json

from dataset import register_coco, register_voc
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow_w_unlabeled
from modeling.generalized_stac_rcnn import ResNetFPNModel, ResNetC4Model
from utils.stac_helper import PathLog

from FasterRCNN.eval import EvalCallback

try:
  import horovod.tensorflow as hvd
except ImportError:
  pass

if __name__ == '__main__':
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--load',
      help='Load a model to start training from. It overwrites BACKBONE.WEIGHTS'
  )
  parser.add_argument(
      '--simple_path',
      action='store_true',
      help='Use direct path rather than automatically generated long one with parameter string'
  )
  parser.add_argument(
      '--resume', action='store_true', help='Resume from the latest checkpoint')
  parser.add_argument(
      '--logdir',
      help='Log directory. Will remove the old one if already exists.',
      default='train_log/maskrcnn')
  parser.add_argument(
      '--pseudo_path', help='pseudo path to save pseudo_data.py')
  parser.add_argument(
      '--config',
      help='A list of KEY=VALUE to overwrite those defined in config.py',
      nargs='+')

  if get_tf_version_tuple() < (1, 6):
    # https://github.com/tensorflow/tensorflow/issues/14657
    logger.warn(
        "TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky."
    )

  args = parser.parse_args()
  if args.config:
    sfx = cfg.update_args(args.config)
    if not args.simple_path:
      args.logdir = os.path.join(args.logdir, sfx)
  #VOC config
  if 'VOC' in cfg.DATA.TRAIN[0]:
    voc_config = [
        'TRAIN.BASE_LR=0.001', 'TRAIN.WARMUP_INIT_LR=0.001',
        'TRAIN.LR_SCHEDULE=[7500,40000]', 'RPN.ANCHOR_SIZES=(8,16,32)',
        'PREPROC.TRAIN_SHORT_EDGE_SIZE=[600, 600]',
        'PREPROC.TEST_SHORT_EDGE_SIZE=600', 'TEST.FRCNN_NMS_THRESH=0.3',
        'TEST.RESULT_SCORE_THRESH=0.0001', 'PREPROC.MAX_SIZE=1000',
        'FRCNN.BATCH_PER_IM=256', 'TRAIN.EVAL_PERIOD=10', 'DATA.NUM_WORKERS=32'
    ]
    cfg.update_args(voc_config)
    cfg.update_args(args.config)
  try:
    register_voc(cfg.DATA.BASEDIR)  # add VOC datasets to the registry
  except:
    logger.warning('VOC does not find!')
  register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry

  # Setup logging ...
  is_horovod = cfg.TRAINER == 'horovod'
  if is_horovod:
    hvd.init()
  if not is_horovod or hvd.rank() == 0:
    logger.set_logger_dir(args.logdir, 'b' if not args.resume else 'k')
  logger.info('Environment Information:\n' + collect_env_info())

  finalize_configs(is_training=True)
  assert cfg.MODE_MASK is False, 'Does not support MaskRCNN'

  # Create model
  MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

  # Compute the training schedule from the number of GPUs ...
  stepnum = cfg.TRAIN.STEPS_PER_EPOCH
  # warmup is step based, lr is epoch based
  init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
  warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
  warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
  lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

  factor = 8. / cfg.TRAIN.NUM_GPUS
  for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
    mult = cfg.TRAIN.GAMMA**(idx + 1)
    lr_schedule.append((steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
  logger.info('Warm Up Schedule (steps, value): ' + str(warmup_schedule))
  logger.info('LR Schedule (epochs, value): ' + str(lr_schedule))

  pseudo_path = os.path.join(args.pseudo_path, 'pseudo_data.npy')
  train_dataflow = get_train_dataflow_w_unlabeled(pseudo_path)

  # This is what's commonly referred to as "epochs"
  total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
  logger.info(
      'Total passes of the training set is: {:.5g}'.format(total_passes))

  # Create callbacks ...
  callbacks = [
      PeriodicCallback(
          ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
          every_k_epochs=cfg.TRAIN.CHECKPOINT_PERIOD),
      # linear warmup
      ScheduledHyperParamSetter(
          'learning_rate', warmup_schedule, interp='linear', step_based=True),
      ScheduledHyperParamSetter('learning_rate', lr_schedule),
      GPUMemoryTracker(),
      HostMemoryTracker(),
      ThroughputTracker(samples_per_step=cfg.TRAIN.NUM_GPUS),
      EstimatedTimeLeft(median=True),
      SessionRunTimeout(60000),  # 1 minute timeout
      GPUUtilizationTracker(),
      PathLog(args.logdir + '\nPseudo path: {}'.format(pseudo_path))
  ]
  # if the main model is starts from a meaningful point, we eval at start.
  if cfg.TRAIN.EVAL_PERIOD > 0:
    callbacks.extend([
        EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
        for dataset in cfg.DATA.VAL
    ])

  if is_horovod and hvd.rank() > 0:
    session_init = None
  else:
    if args.resume:

      last_ckpt = tf.train.latest_checkpoint(args.logdir)
      assert last_ckpt is not None, 'Did not find latest checkpoint from ' + args.logdir
      logger.info('Find latest checkpoint {} to resume'.format(last_ckpt))
      session_init = SmartInit(last_ckpt)
      fname = os.path.join(args.logdir, JSONWriter.FILENAME)
      assert os.path.isfile(
          fname
      ), 'Can not find stats.json to load lastest checkpoint from ' + args.logdir
      # stats = json.load(open(fname))
    elif args.load:
      # ignore mismatched values, so you can `--load` a model for fine-tuning
      session_init = SmartInit(args.load, ignore_mismatch=True)
    else:
      session_init = SmartInit(cfg.BACKBONE.WEIGHTS)

  if args.resume:
    traincfg = AutoResumeTrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
    )
  else:
    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
        starting_epoch=cfg.TRAIN.STARTING_EPOCH)

  if is_horovod:
    trainer = HorovodTrainer(average=False)
  else:
    # nccl mode appears faster than cpu mode
    trainer = SyncMultiGPUTrainerReplicated(
        cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
  launch_train_with_config(traincfg, trainer)
