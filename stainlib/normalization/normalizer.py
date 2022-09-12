import numpy as np
from stainlib.extraction.macenko_stain_extractor import MacenkoStainExtractor
from stainlib.extraction.vahadane_stain_extractor import VahadaneStainExtractor
from stainlib.utils.stain_utils import normalize_matrix_rows, convert_RGB_to_OD, LuminosityThresholdTissueLocator

#from stainlib.normalization.reinhard_stain_normalizer import ReinhardStainNormalizer
from stainlib.utils.stain_utils import (ABCStainExtractor,
                                        LuminosityThresholdTissueLocator,
                                        convert_OD_to_RGB, convert_RGB_to_OD,
                                        get_concentrations, get_mean_std,
                                        is_uint8_image, lab_split, merge_back,
                                        normalize_matrix_rows,
                                        standardize_brightness)


class ExtractiveStainNormalizer(object):
    def __init__(self, method):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        #elif method.lower() == 'reinhard':
        #    self.extractor = ReinhardStainNormalizer
        else:
            raise Exception('Method not recognized.')

    def fit(self, target):
        """
        Fit to a target image.
        :param target: Image RGB uint8.
        :return:
        """

        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
        #self.stain_matrix_target_RGB = convert_OD_to_RGB(self.stain_matrix_target)  # useful to visualize.

    def transform(self, I):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :return:
        """
        stain_matrix_source = self.extractor.get_stain_matrix(I)
        source_concentrations = get_concentrations(I, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= (self.maxC_target / maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)



class ReinhardStainNormalizer(object):
    """
    Normalize a patch stain to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
    """

    def __init__(self, target_means=0,target_stds=0):
        self.target_means = target_means
        self.target_stds = target_stds

    def fit(self, target):
        target =  standardize_brightness(target)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I, mask_background = False, luminosity_threshold=0.8):
        """
        Transform the image I using Reinhard histogram matching.
        :param I: Image RGB uint8.
        :param mask_background: False (default) To include (or not) the background in the normalization boolean.
        :return: I_transformed: Image transformed uint8
        """

        I = standardize_brightness(I)
        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]

        if mask_background:
            tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I, luminosity_threshold=luminosity_threshold).reshape((-1,))
            tissue_mask = tissue_mask.reshape((I.shape[0], I.shape[1]))
            background = np.array(~tissue_mask*254).astype(np.uint8)
            norm1,norm2,norm3 = np.multiply(tissue_mask,norm1), np.multiply(tissue_mask,norm2), np.multiply(tissue_mask,norm3)
            I_transformed = merge_back(background+norm1, norm2, norm3) 
        else:
            I_transformed = merge_back(norm1, norm2, norm3) 
        
        return I_transformed
