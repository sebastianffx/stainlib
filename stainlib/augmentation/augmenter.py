"""
Parts of this code were provided by RadboudUMC for the ExaMode consortium.
See README file for credits.
"""
import skimage.color
from skimage.color import rgb2gray
import numpy as np
import numpy as np
import copy
import pylab as plt
from PIL import Image
from stainlib.utils.excepts import  InvalidRangeError
from stainlib.utils.stain_utils import is_uint8_image
from stainlib.extraction.macenko_stain_extractor import MacenkoStainExtractor
from stainlib.extraction.vahadane_stain_extractor import VahadaneStainExtractor
from stainlib.utils.stain_utils import get_concentrations, LuminosityThresholdTissueLocator
#

class AugmenterBase(object):
    """Base class for patch augmentation."""

    def __init__(self, keyword):
        """
        Args:
            keyword (str): Short name for the transformation.
        """
        super().__init__()
        self._keyword = keyword

    @property
    def keyword(self):
        """
        Get the keyword for the augmenter.

        Returns:
            str: Keyword.
        """

        return self._keyword

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.

        Args:
            target_shapes (dict): Target output shape per level.

        Returns:
            (dict): Required input shape per level.
        """

        # By default the output shapes match the input shapes.
        return target_shapes

    def transform(self, patch):
        """
        Transform the given patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        pass

    def randomize(self):
        """Randomize the parameters of the augmenter."""
        pass

class ColorAugmenterBase(AugmenterBase):
    """Base class for color patch augmentation."""

    def __init__(self, keyword):
        """
        Initialize the object.

        Args:
            keyword (str): Short name for the transformation.
        """

        # Initialize the base class.
        super().__init__(keyword=keyword)

class HedColorAugmenter(ColorAugmenterBase):
    """Apply color correction in HED color space on the RGB patch."""

    def __init__(
        self,
        haematoxylin_sigma_range,
        haematoxylin_bias_range,
        eosin_sigma_range,
        eosin_bias_range,
        dab_sigma_range,
        dab_bias_range,
        cutoff_range,
    ):
        """
        Initialize the object. For each channel the augmented value is calculated as value = value * sigma + bias

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_bias_range (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented. Values from the [0.0, 1.0] range. The RGB channel values are from the same range.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Initialize base class.
        super().__init__(keyword="hed_color")

        # Initialize members.
        self._sigma_ranges = None  # Configured sigma ranges for H, E, and D channels.
        self._bias_ranges = None  # Configured bias ranges for H, E, and D channels.
        self._cutoff_range = None  # Cutoff interval.
        self._sigmas = None  # Randomized sigmas for H, E, and D channels.
        self._biases = None  # Randomized biases for H, E, and D channels.

        # Save configuration.
        self._setsigmaranges(
            haematoxylin_sigma_range=haematoxylin_sigma_range,
            eosin_sigma_range=eosin_sigma_range,
            dab_sigma_range=dab_sigma_range,
        )
        self._setbiasranges(
            haematoxylin_bias_range=haematoxylin_bias_range,
            eosin_bias_range=eosin_bias_range,
            dab_bias_range=dab_bias_range,
        )
        self._setcutoffrange(cutoff_range=cutoff_range)

    def _setsigmaranges(self, haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range):
        """
        Set the sigma intervals.

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        if haematoxylin_sigma_range is not None:
            if (
                len(haematoxylin_sigma_range) != 2
                or haematoxylin_sigma_range[1] < haematoxylin_sigma_range[0]
                or haematoxylin_sigma_range[0] < -1.0
                or 1.0 < haematoxylin_sigma_range[1]
            ):
                raise InvalidRangeError('Haematoxylin Sigma', haematoxylin_sigma_range)

        if eosin_sigma_range is not None:
            if (
                len(eosin_sigma_range) != 2
                or eosin_sigma_range[1] < eosin_sigma_range[0]
                or eosin_sigma_range[0] < -1.0
                or 1.0 < eosin_sigma_range[1]
            ):
                raise InvalidRangeError('Eosin Sigma', eosin_sigma_range)

        if dab_sigma_range is not None:
            if (
                len(dab_sigma_range) != 2
                or dab_sigma_range[1] < dab_sigma_range[0]
                or dab_sigma_range[0] < -1.0
                or 1.0 < dab_sigma_range[1]
            ):
                raise InvalidRangeError('Dab Sigma', dab_sigma_range)

        # Store the settings.
        self._sigma_ranges = [
            haematoxylin_sigma_range,
            eosin_sigma_range,
            dab_sigma_range,
        ]

        self._sigmas = [
            haematoxylin_sigma_range[0] if haematoxylin_sigma_range is not None else 0.0,
            eosin_sigma_range[0] if eosin_sigma_range is not None else 0.0,
            dab_sigma_range[0] if dab_sigma_range is not None else 0.0,
        ]

    def _setbiasranges(self, haematoxylin_bias_range, eosin_bias_range, dab_bias_range):
        """
        Set the bias intervals.

        Args:
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel.
            dab_bias_range (tuple, None): Bias range for the DAB channel.

        Raises:
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        if haematoxylin_bias_range is not None:
            if (
                len(haematoxylin_bias_range) != 2
                or haematoxylin_bias_range[1] < haematoxylin_bias_range[0]
                or haematoxylin_bias_range[0] < -1.0
                or 1.0 < haematoxylin_bias_range[1]
            ):
                raise InvalidRangeError('Haematoxylin Bias', haematoxylin_bias_range)

        if eosin_bias_range is not None:
            if (
                len(eosin_bias_range) != 2
                or eosin_bias_range[1] < eosin_bias_range[0]
                or eosin_bias_range[0] < -1.0
                or 1.0 < eosin_bias_range[1]
            ):
                raise InvalidRangeError('Eosin Bias', eosin_bias_range)

        if dab_bias_range is not None:
            if (
                len(dab_bias_range) != 2
                or dab_bias_range[1] < dab_bias_range[0]
                or dab_bias_range[0] < -1.0
                or 1.0 < dab_bias_range[1]
            ):
                raise InvalidRangeError('Dab Bias', dab_bias_range)

        # Store the settings.
        self._bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]

        self._biases = [
            haematoxylin_bias_range[0] if haematoxylin_bias_range is not None else 0.0,
            eosin_bias_range[0] if eosin_bias_range is not None else 0.0,
            dab_bias_range[0] if dab_bias_range is not None else 0.0,
        ]

    def _setcutoffrange(self, cutoff_range):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval will not be augmented.

        Args:
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented.

        Raises:
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Check the interval.
        if cutoff_range is not None:
            if (
                len(cutoff_range) != 2
                or cutoff_range[1] < cutoff_range[0]
                or cutoff_range[0] < 0.0
                or 1.0 < cutoff_range[1]
            ):
                raise InvalidRangeError('Cutoff', cutoff_range)

        # Store the setting.
        self._cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Check if the patch is inside the cutoff values.
        if patch.dtype.kind == "f":
            patch_mean = np.mean(a=patch)
        else:
            patch_mean = np.mean(a=patch.astype(dtype=np.float32)) / 255.0

        if self._cutoff_range[0] <= patch_mean <= self._cutoff_range[1]:
            # Convert the image patch to HED color coding.
            patch_hed = skimage.color.rgb2hed(rgb=patch)

            # Augment the Haematoxylin channel.
            if self._sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= 1.0 + self._sigmas[0]

            if self._biases[0] != 0.0:
                patch_hed[:, :, 0] += self._biases[0]

            # Augment the Eosin channel.
            if self._sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= 1.0 + self._sigmas[1]

            if self._biases[1] != 0.0:
                patch_hed[:, :, 1] += self._biases[1]

            # Augment the DAB channel.
            if self._sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= 1.0 + self._sigmas[2]

            if self._biases[2] != 0.0:
                patch_hed[:, :, 2] += self._biases[2]

            # Convert back to RGB color coding.
            patch_rgb = skimage.color.hed2rgb(hed=patch_hed)
            patch_transformed = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)

            # Convert back to integral data type if the input was also integral.
            if patch.dtype.kind != "f":
                patch_transformed *= 255.0
                patch_transformed = patch_transformed.astype(dtype=np.uint8)

            return patch_transformed

        else:
            # The image patch is outside the cutoff interval.
            return patch

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize sigma and bias for each channel.
        self._sigmas = [
            np.random.uniform(low=sigma_range[0], high=sigma_range[1], size=None) if sigma_range is not None else 1.0
            for sigma_range in self._sigma_ranges
        ]
        self._biases = [
            np.random.uniform(low=bias_range[0], high=bias_range[1], size=None) if bias_range is not None else 0.0
            for bias_range in self._bias_ranges
        ]

class HedColorAugmenter1(HedColorAugmenter):
    def __init__(self, thresh):
        val = thresh
        bias_val = thresh
        haematoxylin_sigma_range = (-val, val) #(tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
        haematoxylin_bias_range = (-bias_val, bias_val) # (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
        eosin_sigma_range = (-val, val) # (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
        eosin_bias_range = (-bias_val, bias_val) # (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
        dab_sigma_range = (-val, val) # (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
        dab_bias_range = (-bias_val, bias_val) # (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
        cutoff_range = (0.05, 0.95) # (tuple, None) #ignore almost empty patches
        # cutoff_range = (0.0, 1.0)  # (tuple, None):
        super().__init__(haematoxylin_sigma_range, haematoxylin_bias_range,
                          eosin_sigma_range, eosin_bias_range,
                          dab_sigma_range, dab_bias_range, cutoff_range)

class HedLighterColorAugmenter(HedColorAugmenter1):
    def __init__(self):
        super().__init__(0.03)

class HedLightColorAugmenter(HedColorAugmenter1):
    def __init__(self):
        super().__init__(0.1)

class HedStrongColorAugmenter(HedColorAugmenter1):
    def __init__(self):
        super().__init__(1.0)

class GrayscaleAugmentor(object):
    def __init__(self, sigma1=0.2, sigma2=0.2, augment_background=False):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background

    def fit(self, I):
        """
        Fit to an image I.
        :param I:
        :return:
        """
        self.image_shape = I.shape
        self.tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I).ravel()
        self.image  = I
        
    def pop(self):
        """
        Get an augmented version of the fitted image.
        :return:
        """
        alpha = np.random.uniform(1 - 0.2, 1 + 0.2)
        beta = np.random.uniform(-0.2, 0.2)
        grayscale = rgb2gray(self.image)
        grayscale = np.clip((grayscale*alpha) + beta, 0, 1)
        grayscale_threechannels = np.stack([grayscale,grayscale,grayscale], axis=2)
        grayscale_threechannels = np.clip(grayscale_threechannels * 255, 0, 255).astype(np.uint8)
        return grayscale_threechannels

class StainAugmentor(object):

    def __init__(self, method, sigma1=0.2, sigma2=0.2, augment_background=False):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception('Method not recognized.')
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background

    def fit(self, I):
        """
        Fit to an image I.
        :param I:
        :return:
        """
        self.image_shape = I.shape
        self.stain_matrix = self.extractor.get_stain_matrix(I)
        self.source_concentrations = get_concentrations(I, self.stain_matrix)
        self.n_stains = self.source_concentrations.shape[1]
        self.tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I).ravel()

    def pop(self):
        """
        Get an augmented version of the fitted image.
        :return:
        """
        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
        I_augmented = I_augmented.reshape(self.image_shape)
        I_augmented = np.clip(I_augmented, 0, 255).astype(np.uint8)

        return I_augmented