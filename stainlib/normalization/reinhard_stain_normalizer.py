"""
Normalize a patch stain to the target image using the method of:
E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

import numpy as np
from stainlib.utils.stain_utils import ABCStainExtractor, is_uint8_image
from stainlib.utils.stain_utils import normalize_matrix_rows, convert_RGB_to_OD, LuminosityThresholdTissueLocator
from stainlib.utils.stain_utils import standardize_brightness, get_mean_std, lab_split, merge_back
from stainlin.normalization.normalizer import StainNormalizer

class ReinhardStainNormalizer(StainNormalizer):
    @staticmethod
    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        target =  standardize_brightness(target)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        I = standardize_brightness(I)
        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return merge_back(norm1, norm2, norm3)
