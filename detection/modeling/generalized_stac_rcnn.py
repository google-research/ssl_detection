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

"""STAC model class."""

import tensorflow as tf
from tensorpack.utils import logger
import sys

from tensorpack import ModelDesc
from tensorpack.models import GlobalAvgPooling, l2_regularizer, regularize_cost
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

from config import config as cfg
from utils.stac_helper import add_moving_summary_no_nan

from FasterRCNN.modeling import model_frcnn
from FasterRCNN.modeling import model_mrcnn
from FasterRCNN.modeling.backbone import image_preprocess, resnet_c4_backbone, resnet_conv5, resnet_fpn_backbone
from FasterRCNN.modeling.model_box import RPNAnchors, clip_boxes, crop_and_resize, roi_align
from FasterRCNN.modeling.model_cascade import CascadeRCNNHead
from FasterRCNN.modeling.model_fpn import fpn_model, generate_fpn_proposals, multilevel_roi_align, multilevel_rpn_losses
from FasterRCNN.modeling.model_frcnn import (BoxProposals, FastRCNNHead,
                                             fastrcnn_outputs,
                                             fastrcnn_predictions,
                                             sample_fast_rcnn_targets)
from FasterRCNN.modeling.model_mrcnn import maskrcnn_loss, maskrcnn_upXconv_head, unpackbits_masks
from FasterRCNN.modeling.model_rpn import generate_rpn_proposals, rpn_head, rpn_losses
from FasterRCNN.utils import np_box_ops
from FasterRCNN.utils.box_ops import area as tf_area
from FasterRCNN.data import get_all_anchors, get_all_anchors_fpn


class SSLOD(object):
  """Utility functions for SSL Object Detection."""

  def visualize_images(self, image, boxes, name):
    """Visualize images in tensorboard.

    Args:
        images: BxHxWx3
        boxes: Bx4 (x1, y1, x2, y2)
        name: name of tensor
    """
    with tf.name_scope('imgbox'):
      size = tf.shape(image)[-2:]
      h, w = tf.cast(size[0], tf.float32), tf.cast(size[1], tf.float32)
      # convert to [y_min, x_min, y_max, x_max] and normalize
      boxes = tf.stack(
          [boxes[:, 1] / h, boxes[:, 0] / w, boxes[:, 3] / h, boxes[:, 2] / w],
          1)
      # colors = tf.constant([1.0, 0.0, 0.0])
      image = tf.transpose(image, perm=(0, 2, 3, 1))  # -> BxHxWx3

      image_w_box = tf.image.draw_bounding_boxes(image, [boxes])
      minv, maxv = tf.reduce_mean(image_w_box), tf.reduce_max(image_w_box)
      image_w_box = tf.identity((image_w_box - minv) / (maxv - minv), name=name)
      tf.summary.image(name, image_w_box, max_outputs=10)


class GeneralizedRCNN(ModelDesc, SSLOD):

  def preprocess(self, image):
    image = tf.expand_dims(image, 0)
    image = image_preprocess(image, bgr=True)
    return tf.transpose(image, [0, 3, 1, 2])

  def optimizer(self):
    lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
    tf.summary.scalar('learning_rate-summary', lr)

    # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
    if cfg.TRAIN.NUM_GPUS > 1:
      lr = lr / 8.
      opt = tf.train.MomentumOptimizer(lr, 0.9)
      if cfg.TRAIN.NUM_GPUS < 8:
        opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
    else:
      opt = tf.train.MomentumOptimizer(lr, 0.9)
    return opt

  def get_inference_tensor_names(self):
    """
        Returns two lists of tensor names to be used to create an inference
        callable.

        `build_graph` must create tensors of these names when called under
        inference context.

        Returns:
            [str]: input names
            [str]: output names
        """
    out = ['output/boxes', 'output/scores', 'output/labels']
    return ['image'], out

  def forward(self, img, tars, inputs, pseudo_proposals=None):
    features = self.backbone(img)
    anchor_inputs = {k: v for k, v in inputs.items() if k.startswith('anchor_')}
    if pseudo_proposals is not None:
      proposals = pseudo_proposals
      rpn_losses = [
          tf.constant(0, dtype=tf.float32),
          tf.constant(0, dtype=tf.float32)
      ]
    else:
      proposals, _, rpn_losses = self.rpn(img, features, anchor_inputs)
    # vars of roi_head already exists in checkpoints, we want to initialize new one
    head_losses = self.roi_heads(img, features, proposals, tars)

    return head_losses, rpn_losses

  def build_graph(self, *inputs):

    logger.info('-' * 100)
    logger.info('This is the official STAC model (MODE_FPN = {})'.format(
        cfg.MODE_FPN))
    logger.info('-' * 100)

    inputs = dict(zip(self.input_names, inputs))

    image_l = self.preprocess(inputs['image'])  # 1CHW
    if self.training:
      # Semi-supervsiedly train the model
      image_u_strong = self.preprocess(inputs['image_strong'])  # 1CHW
      pseudo_targets = [
          inputs[k]
          for k in ['gt_boxes_strong', 'gt_labels_strong']
          if k in inputs
      ]
      if cfg.TRAIN.NO_PRN_LOSS:
        # [labeled and unlabeled]
        proposals_boxes = [None, BoxProposals(inputs['proposals_boxes_strong'])]
      else:
        proposals_boxes = [None, None]

      self.visualize_images(
          image_u_strong, pseudo_targets[0], name='unlabeled_strong')
      self.visualize_images(image_l, inputs['gt_boxes'], name='labeled')

      # get groundtruth of labeled data
      targets = [inputs[k] for k in ['gt_boxes', 'gt_labels'] if k in inputs]
      gt_boxes_area = tf.reduce_mean(
          tf_area(inputs['gt_boxes']), name='mean_gt_box_area')
      add_moving_summary(gt_boxes_area)

      image_list = [image_l, image_u_strong]
      target_list = [targets, pseudo_targets]
      inputs_strong = {
          k.replace('_strong', ''): v
          for k, v in inputs.items()
          if 'strong' in k
      }
      inputs = {
          k: v
          for k, v in inputs.items()
          if ('strong' not in k and 'weak' not in k)
      }  # for labeled data
      input_list = [inputs, inputs_strong]

      # The image are forwarded one by one. labeled image is the one
      # we need to define which forward is the final branch in order to create specified name of outputs
      head_losses = []
      rpn_losses = []

      for i, (im, tar, inp, pbus) in enumerate(
          zip(image_list, target_list, input_list, proposals_boxes)):
        hl_loss, rl_loss = self.forward(im, tar, inp, pseudo_proposals=pbus)
        head_losses.extend(hl_loss)
        rpn_losses.extend(rl_loss)

      k = len(head_losses) // len(image_list)
      # normalize the loss by number of forward
      head_losses = [a / float(len(image_list)) for a in head_losses]
      rpn_losses = [a / float(len(image_list)) for a in rpn_losses]

      # monitor supervised lossfrom pseudo labels/boxes only
      head_losses_u = head_losses[k:]
      rpn_losses_u = rpn_losses[k:]
      head_cost_u = tf.add_n(head_losses_u, name='fxm/head_cost_u')
      rpn_cost_u = tf.add_n(rpn_losses_u, name='fxm/rpn_cost_u')
      add_moving_summary_no_nan(head_cost_u, name='fxm/head_cost_u')
      add_moving_summary_no_nan(rpn_cost_u, name='fxm/rpn_cost_u')

      # multiply wu to unsupervised loss
      head_losses = head_losses[:k] + [a * cfg.TRAIN.WU for a in head_losses_u]
      rpn_losses = rpn_losses[:k] + [a * cfg.TRAIN.WU for a in rpn_losses_u]

    else:
      targets = [inputs[k] for k in ['gt_boxes', 'gt_labels'] if k in inputs]
      self.forward(image_l, targets, inputs)

    if self.training:
      regex = '.*/W'
      wd_cost = regularize_cost(
          regex, l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')
      assert 'empty' not in wd_cost.name
      total_cost = tf.add_n(rpn_losses + head_losses + [wd_cost], 'total_cost')
      add_moving_summary_no_nan(total_cost, 'total_loss')
      add_moving_summary_no_nan(wd_cost, 'wd_loss')
      return total_cost
    else:
      # Check that the model defines the tensors it declares for inference
      # For existing models, they are defined in "fastrcnn_predictions(name_scope='output')"
      G = tf.get_default_graph()
      ns = G.get_name_scope()
      for name in self.get_inference_tensor_names()[1]:
        try:
          name = '/'.join([ns, name]) if ns else name
          G.get_tensor_by_name(name + ':0')
        except KeyError:
          raise KeyError(
              "Your model does not define the tensor '{}' in inference context."
              .format(name))


class ResNetFPNModel(GeneralizedRCNN):

  def inputs(self):
    ret = [tf.TensorSpec((None, None, 3), tf.float32, 'image')]
    ret.extend([tf.TensorSpec((None, None, 3), tf.float32, 'image_strong')])
    num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
    for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
      ret.extend([
          tf.TensorSpec((None, None, num_anchors), tf.int32,
                        'anchor_labels_lvl{}'.format(k + 2)),
          tf.TensorSpec((None, None, num_anchors, 4), tf.float32,
                        'anchor_boxes_lvl{}'.format(k + 2))
      ])
      ret.extend([
          tf.TensorSpec((None, None, num_anchors), tf.int32,
                        'anchor_labels_lvl{}_strong'.format(k + 2)),
          tf.TensorSpec((None, None, num_anchors, 4), tf.float32,
                        'anchor_boxes_lvl{}_strong'.format(k + 2))
      ])

    ret.extend([
        tf.TensorSpec((None, 4), tf.float32, 'gt_boxes'),
        tf.TensorSpec((None,), tf.int64, 'gt_labels')
    ])  # all > 0
    # gt_boxes and gt_labels may exist and only used for monitoring
    ret.extend([
        tf.TensorSpec((None, 4), tf.float32, 'gt_boxes_strong'),
        tf.TensorSpec((None,), tf.int64, 'gt_labels_strong')
    ])  # all > 0
    ret.extend([tf.TensorSpec((None, 4), tf.float32,
                              'proposals_boxes_strong')])  # all > 0

    return ret

  def slice_feature_and_anchors(self, p23456, anchors):
    for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
      with tf.name_scope('FPN_slice_lvl{}'.format(i)):
        anchors[i] = anchors[i].narrow_to(p23456[i])

  @auto_reuse_variable_scope
  def backbone(self, image):
    c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS)
    p23456 = fpn_model('fpn', c2345)
    return p23456

  @auto_reuse_variable_scope
  def rpn(self, image, features, inputs):
    assert len(cfg.FPN.ANCHOR_SIZES) == len(cfg.FPN.ANCHOR_STRIDES)

    image_shape2d = tf.shape(image)[2:]  # h,w
    all_anchors_fpn = get_all_anchors_fpn(
        strides=cfg.FPN.ANCHOR_STRIDES,
        sizes=cfg.FPN.ANCHOR_SIZES,
        ratios=cfg.RPN.ANCHOR_RATIOS,
        max_size=cfg.PREPROC.MAX_SIZE)
    multilevel_anchors = [
        RPNAnchors(all_anchors_fpn[i],
                   inputs['anchor_labels_lvl{}'.format(i + 2)],
                   inputs['anchor_boxes_lvl{}'.format(i + 2)])
        for i in range(len(all_anchors_fpn))
    ]
    self.slice_feature_and_anchors(features, multilevel_anchors)

    # Multi-Level RPN Proposals
    rpn_outputs = [
        rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
        for pi in features
    ]
    multilevel_label_logits = [k[0] for k in rpn_outputs]
    multilevel_box_logits = [k[1] for k in rpn_outputs]
    multilevel_pred_boxes = [
        anchor.decode_logits(logits)
        for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)
    ]

    proposal_boxes, proposal_scores = generate_fpn_proposals(
        multilevel_pred_boxes, multilevel_label_logits, image_shape2d)

    if self.training:
      losses = multilevel_rpn_losses(multilevel_anchors,
                                     multilevel_label_logits,
                                     multilevel_box_logits)
    else:
      losses = []

    return BoxProposals(proposal_boxes), proposal_scores, losses

  @auto_reuse_variable_scope
  def roi_heads(self, image, features, proposals, targets, training=None):
    # training could overwrite global self.training

    if training is None:
      training = self.training

    image_shape2d = tf.shape(image)[2:]  # h,w
    assert len(features) == 5, 'Features have to be P23456!'
    if len(targets) > 2:
      assert cfg.TRAIN.SAMPLE_BG_BEFORE_MASK
      gt_boxes, gt_labels, gt_boxes_origin, gt_labels_origin, *_ = targets
      if training:
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes_origin,
                                             gt_labels_origin)
    else:
      gt_boxes, gt_labels, *_ = targets
      if training:
        proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes,
                                             gt_labels)

    fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
    if not cfg.FPN.CASCADE:
      roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes,
                                                  7)

      head_feature = fastrcnn_head_func('fastrcnn', roi_feature_fastrcnn)
      fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
          'fastrcnn/outputs', head_feature, cfg.DATA.NUM_CATEGORY)
      fastrcnn_head = FastRCNNHead(
          proposals, fastrcnn_box_logits, fastrcnn_label_logits, gt_boxes,
          tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))
    else:

      def roi_func(boxes):
        return multilevel_roi_align(features[:4], boxes, 7)

      fastrcnn_head = CascadeRCNNHead(proposals, roi_func, fastrcnn_head_func,
                                      (gt_boxes, gt_labels), image_shape2d,
                                      cfg.DATA.NUM_CATEGORY)

    if training:
      all_losses = fastrcnn_head.losses()
      return all_losses
    else:
      decoded_boxes = fastrcnn_head.decoded_output_boxes()
      decoded_boxes = clip_boxes(
          decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
      label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
      final_boxes, final_scores, final_labels = fastrcnn_predictions(
          decoded_boxes, label_scores, name_scope='output')


class ResNetC4Model(GeneralizedRCNN):

  def inputs(self):
    ret = [
        tf.TensorSpec((None, None, 3), tf.float32, 'image'),
        tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR), tf.int32,
                      'anchor_labels'),
        tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR, 4), tf.float32,
                      'anchor_boxes'),
        tf.TensorSpec((None, 4), tf.float32, 'gt_boxes'),
        tf.TensorSpec((None,), tf.int64, 'gt_labels')
    ]
    ret.extend([
        tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR), tf.int32,
                      'anchor_labels_strong'),
        tf.TensorSpec((None, None, cfg.RPN.NUM_ANCHOR, 4), tf.float32,
                      'anchor_boxes_strong')
    ])
    ret.extend([tf.TensorSpec((None, None, 3), tf.float32, 'image_strong')])
    # gt_boxes and gt_labels may exist and only used for monitoring
    ret.extend([
        tf.TensorSpec((None, 4), tf.float32, 'gt_boxes_strong'),
        tf.TensorSpec((None,), tf.int64, 'gt_labels_strong')
    ])  # all > 0
    ret.extend([tf.TensorSpec((None, 4), tf.float32,
                              'proposals_boxes_strong')])  # all > 0

    return ret

  @auto_reuse_variable_scope
  def backbone(self, image):
    return [resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCKS[:3])]

  @auto_reuse_variable_scope
  def rpn(self, image, features, inputs):
    featuremap = features[0]
    rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap,
                                                cfg.RPN.HEAD_DIM,
                                                cfg.RPN.NUM_ANCHOR)
    anchors = RPNAnchors(
        get_all_anchors(
            stride=cfg.RPN.ANCHOR_STRIDE,
            sizes=cfg.RPN.ANCHOR_SIZES,
            ratios=cfg.RPN.ANCHOR_RATIOS,
            max_size=cfg.PREPROC.MAX_SIZE), inputs['anchor_labels'],
        inputs['anchor_boxes'])
    anchors = anchors.narrow_to(featuremap)

    image_shape2d = tf.shape(image)[2:]  # h,w
    pred_boxes_decoded = anchors.decode_logits(
        rpn_box_logits)  # fHxfWxNAx4, floatbox
    proposal_boxes, proposal_scores = generate_rpn_proposals(
        tf.reshape(pred_boxes_decoded, [-1, 4]),
        tf.reshape(rpn_label_logits,
                   [-1]), image_shape2d, cfg.RPN.TRAIN_PRE_NMS_TOPK
        if self.training else cfg.RPN.TEST_PRE_NMS_TOPK,
        cfg.RPN.TRAIN_POST_NMS_TOPK
        if self.training else cfg.RPN.TEST_POST_NMS_TOPK)

    if self.training:
      losses = rpn_losses(anchors.gt_labels, anchors.encoded_gt_boxes(),
                          rpn_label_logits, rpn_box_logits)
    else:
      losses = []

    return BoxProposals(proposal_boxes), proposal_scores, losses

  @auto_reuse_variable_scope
  def roi_heads(self, image, features, proposals, targets, training=None):

    if training is None:
      training = self.training

    image_shape2d = tf.shape(image)[2:]  # h,w
    featuremap = features[0]

    gt_boxes, gt_labels, *_ = targets

    if training:
      # sample proposal boxes in training
      proposals = sample_fast_rcnn_targets(proposals.boxes, gt_boxes, gt_labels)
    # The boxes to be used to crop RoIs.
    # Use all proposal boxes in inference

    boxes_on_featuremap = proposals.boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
    roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

    feature_fastrcnn = resnet_conv5(
        roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCKS[-1])  # nxcx7x7
    # Keep C5 feature to be shared with mask branch
    feature_gap = GlobalAvgPooling(
        'gap', feature_fastrcnn, data_format='channels_first')
    fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
        'fastrcnn', feature_gap, cfg.DATA.NUM_CATEGORY)

    fastrcnn_head = FastRCNNHead(
        proposals, fastrcnn_box_logits, fastrcnn_label_logits, gt_boxes,
        tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32))

    if training:
      all_losses = fastrcnn_head.losses()
      return all_losses
    else:
      decoded_boxes = fastrcnn_head.decoded_output_boxes()
      decoded_boxes = clip_boxes(
          decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
      label_scores = fastrcnn_head.output_scores(name='fastrcnn_all_scores')
      final_boxes, final_scores, final_labels = fastrcnn_predictions(
          decoded_boxes, label_scores, name_scope='output')
