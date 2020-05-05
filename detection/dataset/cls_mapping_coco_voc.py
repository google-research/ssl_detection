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

"""Generate coco class mapping to voc 20 class.

The resulting mapping dict will be hard-coded in coco.py

python dataset/cls_mapping_coco_voc.py
"""

voc_class_names = [
    "motorbike", "dog", "person", "horse", "sofa", "bicycle", "cow", "boat",
    "train", "car", "bird", "cat", "chair", "pottedplant", "sheep", "aeroplane",
    "bottle", "bus", "diningtable", "tvmonitor"
]

coco_class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


identity_name_mapping = {
    # voc to coco name mapping of same class
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "sofa": "couch",
    "pottedplant": "potted plant",
    "tvmonitor": "tv",
    "diningtable": "dining table",

}


COCO_id_to_category_id = {
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    27: 25,
    28: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    65: 60,
    67: 61,
    70: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    82: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    90: 80
}  # noqa

category_id_to_COCO_id = {v: k for k, v in COCO_id_to_category_id.items()}

mapping = {}
for i, name in enumerate(voc_class_names):
  index = coco_class_names.index(identity_name_mapping.get(name, name)) + 1
  coco_index = category_id_to_COCO_id.get(index, index)
  mapping[coco_index] = i + 1

print(
    mapping
)  # {64: 12, 1: 1, 67: 17, 3: 9, 4: 7, 5: 3, 6: 4, 7: 15, 72: 13, 9: 8, 44: 16, 2: 19, 16: 6, 17: 14, 18: 18, 19: 10, 20: 20, 21: 5, 62: 2, 63: 11}
