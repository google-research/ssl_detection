## Environment setting

```bash
export PRJROOT=/path/to/your/project/directory/STAC
export DATAROOT=/path/to/your/dataroot
export RESULTDIR=/path/to/save/model
export COCODIR=$DATAROOT/coco
export VOCDIR=$DATAROOT/voc
export PYTHONPATH=$PYTHONPATH:${PRJROOT}/third_party/FasterRCNN:${PRJROOT}/third_party/auto_augment:${PRJROOT}/third_party/tensorpack
```

## Prepare data

### Download COCO data

```bash
mkdir -p ${COCODIR}
cd ${COCODIR}

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/unlabeled2017.zip

unzip annotations_trainval2017.zip -d .
unzip -q train2017.zip -d .
unzip -q val2017.zip -d .
unzip -q unlabeled2017.zip -d .

# resulting format
# ${COCODIR}
#   - train2017
#     - xxx.jpg
#   - val2017
#     - xxx.jpg
#   - unlabled2017
#     - xxx.jpg
#   - annotations
#     - xxx.json
#     - ...
```

### Download VOC data

```bash
mkdir -p ${VOCDIR}
cd ${VOCDIR}

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar

# resulting format
# ${VOCDIR}
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

### Generate labeled and unlabeled splits with different proportions of labeled data

```bash
cd ${PRJROOT}/prepare_datasets

# Format:
#  labeled split - <datasetname>.<seed>@<percent_of_labeld>
#  unlabeled split - <datasetname>.<seed>@<percent_of_labeld>-unlabeled
for seed in 1 2 3 4 5; do
  for percent in 1 2 5 10 20; do
    python3 prepare_coco_data.py --percent $percent --seed $seed &
  done
done

```

### Download JSON files for unlabeled images of COCO data and PASCAL VOC data

```bash
cd ${DATAROOT}

wget https://storage.cloud.google.com/gresearch/ssl_detection/STAC_JSON.tar
tar -xf STAC_JSON.tar.gz

# coco/annotations/instances_unlabeled2017.json
# coco/annotations/semi_supervised/instances_unlabeledtrainval20class.json
# voc/VOCdevkit/VOC2007/instances_diff_test.json
# voc/VOCdevkit/VOC2007/instances_diff_trainval.json
# voc/VOCdevkit/VOC2007/instances_test.json
# voc/VOCdevkit/VOC2007/instances_trainval.json
# voc/VOCdevkit/VOC2012/instances_diff_trainval.json
# voc/VOCdevkit/VOC2012/instances_trainval.json

```


