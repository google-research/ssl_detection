# Lint as: python3
"""TODO(zizhaoz): DO NOT SUBMIT without one-line documentation for custom.

TODO(zizhaoz): DO NOT SUBMIT without a detailed description of custom.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np

FLAGS = flags.FLAGS


def find_bg_and_fg_proposals(scores, ratios=(0.1, 0.1)):
  """Find top kb% and bottom kf% boxes as background and forground indices.

  Args:
    ratios: a tuple of (float, float) or (int, int). Float represents ratios and
      int represents actual numbers.
    scores: Nx81: 0 is background
  """
  if type(ratios[0]) is int:
    n1, n2 = ratios
    assert type(ratios[1]) is int
  else:
    assert type(ratios[1]) is float
    n1 = int(scores.shape[0] * ratios[0])
    n2 = int(scores.shape[0] * ratios[1])

  bg_scores = scores[:, 0]
  sorted_ind = np.argsort(bg_scores)[::-1]  # top
  bg_ind = sorted_ind[:n1]  # take the top k background regions
  fg_ind = sorted_ind[-n2:]  # take the top k background regions

  return bg_ind, fg_ind
