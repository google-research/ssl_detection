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

# DATASET='coco_train2017.1@10'
# UNLABELED_DATASET='${DATASET}-unlabeled'
# CKPT_PATH=result/${DATASET}
# PSEUDO_PATH=${CKPT_PATH}/PSEUDO_DATA
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Check pseudo path
if [ ! -d ${PSEUDO_PATH} ]; then
    mkdir -p ${PSEUDO_PATH}
fi

# Evaluate the model for sanity check
python3 predict.py \
    --evaluate ${PSEUDO_PATH}/eval.json \
    --load "${CKPT_PATH}"/model-170000 \
    --config \
    DATA.BASEDIR=${COCODIR} \
    DATA.TRAIN="('${UNLABELED_DATASET}',)"

# Extract pseudo label
python3 predict.py \
    --predict_unlabeled ${PSEUDO_PATH} \
    --load "${CKPT_PATH}"/model-170000 \
    --config \
    DATA.BASEDIR=${COCODIR} \
    DATA.TRAIN="('${UNLABELED_DATASET}',)" \
    EVAL.PSEUDO_INFERENCE=True

echo "Pseudo Label generation is done"
echo ${PSEUDO_PATH}
echo "Now start training STAC"
