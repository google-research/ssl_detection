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

# --simple_path makes train_log/${DATASET}/${EXPNAME} as exact location to save
# DATASET='coco_train2017.1@10'
# CKPT_PATH=result/${DATASET}
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 train_stg1.py \
    --logdir ${CKPT_PATH} --simple_path --config \
    BACKBONE.WEIGHTS=${COCODIR}/ImageNet-R50-AlignPadding.npz \
    DATA.BASEDIR=${COCODIR} \
    DATA.TRAIN="('${DATASET}',)" \
    MODE_MASK=False \
    FRCNN.BATCH_PER_IM=512 \
    PREPROC.TRAIN_SHORT_EDGE_SIZE="[500,800]" \
    TRAIN.EVAL_PERIOD=20 \
    TRAIN.AUGTYPE_LAB='default' \
    $@
