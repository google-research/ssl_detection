## Environment setting

- Make sure you are in STAC env3 environment.

```bash
export PRJROOT=/path/to/your/project/directory/STAC
export DATAROOT=/path/to/your/dataroot
export COCODIR=$DATAROOT/coco
export VOCDIR=$DATAROOT/voc
export PYTHONPATH=$PYTHONPATH:${PRJROOT}/object_detection/FasterRCNN:${PRJROOT}
```

## Download data

```bash
mkdir -p $VOCDIR
cd $VOCDIR

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar

# resulting format
# $VOCDIR
#   - VOCdevkit
#     - VOC2007
#       - Annotations
#       - JPEGImages
#       - ...
#     - VOC2012
#       - Annotations
#       - JPEGImages
#       - ...
```

## Generate labeled and unlabeled splits

- [Download format converter from pascal voc to coco.](https://github.com/CivilNet/Gemfield/blob/master/src/python/pascal_voc_xml2json/pascal_voc_xml2json.py)

```bash
cd ${PRJROOT}/prepare_datasets

wget https://raw.githubusercontent.com/CivilNet/Gemfield/master/src/python/pascal_voc_xml2json/pascal_voc_xml2json.py
```

- Generate json files of pascal voc labeled and unlabeled data.

```bash
cd ${PRJROOT}/prepare_datasets

# Format (TODO):
#  labeled split - <datasetname>.<seed>@<percent_of_labeld>
#  unlabeled split - <datasetname>.<seed>@<percent_of_labeld>-unlabeled
python3 prepare_voc_data.py --data_dir $VOCDIR
```
- Generate json file of pascal voc unlabeled data from coco labeled data with overlapping objects. ([TODO]: Chun-liang?)
- See [here](prepare_datasets/coco_instruction.md) to download coco data.

```bash
cd ${PRJROOT}/prepare_datasets

# Format (TODO):

python prepare_voc_data_from_coco.py
```


REMOVE below

## Train FRCNN on voc
```bash
DATASET='VOC2007/instances_trainval'
TESTSET='VOC2007/instances_test'
CUDA_VISIBLE_DEVICES=0 python train.py --logdir train_log/${DATASET}/f50-fpn --config \
    BACKBONE.WEIGHTS=./ImageNet-R50-AlignPadding.npz \
    DATA.BASEDIR=${VOCDIR}/VOCdevkit \
    DATA.TRAIN="('${DATASET}',)" \
    DATA.VAL="('${TESTSET}',)"
    MODE_MASK=False
```
