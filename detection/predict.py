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
"""Inference entrance."""
# pylint: disable=g-multiple-import
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
# pylint: disable=unused-variable
import argparse
import collections
import itertools
import numpy as np
import os
import shutil
import cv2
import tqdm
import deepdish as dd
import json

import tensorflow as tf
from tensorflow.python.framework import test_util

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import register_coco, register_voc
from config import config as cfg
from config import finalize_configs

from data import get_eval_unlabeled_dataflow

# from third_party
from FasterRCNN.dataset import DatasetRegistry
from FasterRCNN.data import get_eval_dataflow
from FasterRCNN.eval import DetectionResult, multithread_predict_dataflow, predict_image
from FasterRCNN.modeling.generalized_rcnn import ResNetFPNModel, ResNetC4Model
from FasterRCNN.viz import draw_final_outputs, draw_predictions
from FasterRCNN.utils import custom
from FasterRCNN.predict import do_evaluate, do_predict, do_visualize


def predict_unlabeled(model,
                      model_path,
                      nr_visualize=100,
                      output_dir='output_patch_samples'):
  """Predict the pseudo label information of unlabeled data."""

  assert cfg.EVAL.PSEUDO_INFERENCE, 'set cfg.EVAL.PSEUDO_INFERENCE=True'
  df, dataset_size = get_eval_unlabeled_dataflow(
      cfg.DATA.TRAIN, return_size=True)
  df.reset_state()
  predcfg = PredictConfig(
      model=model,
      session_init=SmartInit(model_path),
      input_names=['image'],  # ['image', 'gt_boxes', 'gt_labels'],
      output_names=[
          'generate_{}_proposals/boxes'.format(
              'fpn' if cfg.MODE_FPN else 'rpn'),
          'generate_{}_proposals/scores'.format(
              'fpn' if cfg.MODE_FPN else 'rpn'),
          'fastrcnn_all_scores',
          'output/boxes',
          'output/scores',  # score of the labels
          'output/labels',
      ])
  pred = OfflinePredictor(predcfg)

  if os.path.isdir(output_dir):
    if os.path.isfile(os.path.join(output_dir, 'pseudo_data.npy')):
      os.remove(os.path.join(output_dir, 'pseudo_data.npy'))
    if not os.path.isdir(os.path.join(output_dir, 'vis')):
      os.makedirs(os.path.join(output_dir, 'vis'))
    else:
      shutil.rmtree(os.path.join(output_dir, 'vis'))
      fs.mkdir_p(output_dir + '/vis')
  else:
    fs.mkdir_p(output_dir)
    fs.mkdir_p(output_dir + '/vis')
  logger.warning('-' * 100)
  logger.warning('Write to {}'.format(output_dir))
  logger.warning('-' * 100)

  with tqdm.tqdm(total=nr_visualize) as pbar:
    for idx, dp in itertools.islice(enumerate(df), nr_visualize):
      img, img_id = dp  # dp['image'], dp['img_id']
      rpn_boxes, rpn_scores, all_scores, \
          final_boxes, final_scores, final_labels = pred(img)
      outs = {
          'proposals_boxes': rpn_boxes,  # (?,4)
          'proposals_scores': rpn_scores,  # (?,)
          'boxes': final_boxes,
          'scores': final_scores,
          'labels': final_labels
      }
      ratios = [10, 10]  # [top 20% as background, bottom 20% as background]
      bg_ind, fg_ind = custom.find_bg_and_fg_proposals(
          all_scores, ratios=ratios)

      bg_viz = draw_predictions(img, rpn_boxes[bg_ind], all_scores[bg_ind])

      fg_viz = draw_predictions(img, rpn_boxes[fg_ind], all_scores[fg_ind])

      results = [
          DetectionResult(*args)
          for args in zip(final_boxes, final_scores, final_labels, [None] *
                          len(final_labels))
      ]
      final_viz = draw_final_outputs(img, results)

      viz = tpviz.stack_patches([bg_viz, fg_viz, final_viz], 2, 2)

      if os.environ.get('DISPLAY', None):
        tpviz.interactive_imshow(viz)
      assert cv2.imwrite('{}/vis/{:03d}.png'.format(output_dir, idx), viz)
      pbar.update()
  logger.info('Write {} samples to {}'.format(nr_visualize, output_dir))

  ## Parallel inference the whole unlabled data
  pseudo_preds = collections.defaultdict(list)

  num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
  graph_funcs = MultiTowerOfflinePredictor(predcfg, list(
      range(num_tower))).get_predictors()
  dataflows = [
      get_eval_unlabeled_dataflow(
          cfg.DATA.TRAIN, shard=k, num_shards=num_tower)
      for k in range(num_tower)
  ]

  all_results = multithread_predict_dataflow(dataflows, graph_funcs)

  for id, result in tqdm.tqdm(enumerate(all_results)):
    img_id = result['image_id']
    outs = {
        'proposals_boxes': result['proposal_box'].astype(np.float16),  # (?,4)
        'proposals_scores': result['proposal_score'].astype(np.float16),  # (?,)
        # 'frcnn_all_scores': result['frcnn_score'].astype(np.float16),
        'boxes': result['bbox'].astype(np.float16),  # (?,4)
        'scores': result['score'].astype(np.float16),  # (?,)
        'labels':
            result['category_id'].astype(np.float16)  # (?,)
    }
    pseudo_preds[img_id] = outs
  logger.warn('Writing to {}'.format(
      os.path.join(output_dir, 'pseudo_data.npy')))
  try:
    dd.io.save(os.path.join(output_dir, 'pseudo_data.npy'), pseudo_preds)
  except RuntimeError:
    logger.error('Save failed. Check reasons manually...')


def do_evaluate_unlabeled(pred_config, output_file, reuse=True):
  """Evaluate unlabled data."""

  for i, dataset in enumerate(cfg.DATA.VAL):
    output = output_file + '-' + dataset
    if not os.path.isfile(output) or not reuse:
      if i == 0:
        num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
        graph_funcs = MultiTowerOfflinePredictor(
            pred_config, list(range(num_tower))).get_predictors()
      logger.info('Evaluating {} ...'.format(dataset))
      dataflows = [
          get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
          for k in range(num_tower)
      ]
      all_results = multithread_predict_dataflow(dataflows, graph_funcs)
      eval_metrics = DatasetRegistry.get(dataset).eval_inference_results2(
          all_results, output, threshold=cfg.TRAIN.CONFIDENCE)
    else:
      all_results = json.load(open(output, 'r'))
      eval_metrics = DatasetRegistry.get(dataset).eval_inference_results2(
          all_results, output, threshold=cfg.TRAIN.CONFIDENCE, metric_only=True)

    with open(output + '_cocometric.json', 'w') as f:
      json.dump(eval_metrics, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--load', help='load a model for evaluation.', required=True)
  parser.add_argument(
      '--visualize', action='store_true', help='visualize intermediate results')
  parser.add_argument(
      '--predict_unlabeled', help='visualize intermediate results')
  parser.add_argument('--eval_unlabeled', help='visualize intermediate results')
  parser.add_argument(
      '--evaluate',
      help='Run evaluation. '
      'This argument is the path to the output json evaluation file')
  parser.add_argument(
      '--predict',
      help='Run prediction on a given image. '
      'This argument is the path to the input image file',
      nargs='+')
  parser.add_argument(
      '--benchmark',
      action='store_true',
      help='Benchmark the speed of the model + postprocessing')
  parser.add_argument(
      '--config',
      help='A list of KEY=VALUE to overwrite those defined in config.py',
      nargs='+')
  parser.add_argument('--output-pb', help='Save a model to .pb')
  parser.add_argument('--output-serving', help='Save a model to serving file')

  args = parser.parse_args()
  if args.config:
    cfg.update_args(args.config)
  try:
    register_voc(cfg.DATA.BASEDIR)  # add VOC datasets to the registry
  except NotImplementedError:
    logger.warning('VOC does not find!')
  register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
  MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

  if not tf.test.is_gpu_available():
    assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
        'Inference requires either GPU support or MKL support!'
  assert args.load
  finalize_configs(is_training=False)

  if args.predict or args.visualize:
    cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

  # let the output has the same path logic as checkpoint
  if args.predict_unlabeled:
    output_dir = args.predict_unlabeled
    predict_unlabeled(MODEL, args.load, output_dir=output_dir)

  if args.visualize:
    do_visualize(MODEL, args.load, output_dir=output_dir)
  else:
    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    if args.output_pb:
      ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
    elif args.output_serving:
      ModelExporter(predcfg).export_serving(args.output_serving, optimize=False)

    if args.predict:
      predictor = OfflinePredictor(predcfg)
      for image_file in args.predict:
        do_predict(predictor, image_file)
    elif args.evaluate:
      assert args.evaluate.endswith('.json'), args.evaluate
      do_evaluate(predcfg, args.evaluate)
    elif args.eval_unlabeled:
      assert args.eval_unlabeled.endswith('.json'), args.eval_unlabeled
      do_evaluate_unlabeled(predcfg, args.eval_unlabeled)
    elif args.benchmark:
      df = get_eval_dataflow(cfg.DATA.VAL[0])
      df.reset_state()
      predictor = OfflinePredictor(predcfg)
      for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
        predict_image(img[0], predictor)
