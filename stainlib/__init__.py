"""
Sebastian Otálora and the ExaMode consortium.
This python3 library contains several methods for augmenting and normalizing image patches from whole slide images.
Many of the functions were refactored from the StainTools library: https://github.com/Peter554/StainTools 
Other methods come from previously published articles. If you find the library useful, consider citing the following articles:
1) Khan, Amjad, et al. "Generalizing convolution neural networks on stain color heterogeneous data for computational pathology." 
Medical Imaging 2020: Digital Pathology. Vol. 11320. International Society for Optics and Photonics, 2020. 
2) Tellez, David, et al. "Quantifying the effects of data augmentation and stain color normalization in convolutional neural 
networks for computational pathology." Medical image analysis 58 (2019): 101544.
3) Otálora, Sebastian, et al. "Staining invariant features for improving generalization of deep convolutional 
neural networks in computational pathology." Frontiers in Bioengineering and Biotechnology 7 (2019): 198.
"""

import sys
if sys.version_info[0] < 3:
    raise Exception("Error: You are not running Python 3.")

#Stain extraction classes and modules
from stainlib.extraction.macenko_stain_extractor import MacenkoStainExtractor
from stainlib.extraction.vahadane_stain_extractor import VahadaneStainExtractor

#Stain augmentation classes and modules
from stainlib.augmentation.augmenter import HedLighterColorAugmenter, HedLightColorAugmenter, HedStrongColorAugmenter
from stainlib.augmentation.augmenter import GrayscaleAugmentor

#Stain normalization classes and modules
from stainlib.normalization.normalizer import ExtractiveStainNormalizer
from stainlib.normalization.normalizer import ReinhardStainNormalizer

#from stainlib.utils.luminosity_standardizer import LuminosityStandardizer
#from stainlib.preprocessing.read_image import read_image
#from stainlib.visualization.visualization import *
