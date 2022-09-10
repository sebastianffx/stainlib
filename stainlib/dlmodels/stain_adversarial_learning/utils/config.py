#This file is used to define machine-dependent paths and shared variables across the differents scripts of the experiment
TRAINING_MODE = 'baseline' # Posibilities are: base, color_augmentation, stain_normalization, domain_adversarial
PREFIX_IMGS = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/mitoses_image_data_part_'
PATH_GT = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/mitoses_ground_truth/'

PATH_TEST_INTERNAL_POS = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/test/mitosis/'
PATH_TEST_INTERNAL_NEG = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/test/non_mitosis/'

PATH_TEST_EXTERNAL_POS = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/external/mitosis/'
PATH_TEST_EXTERNAL_NEG = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/external/balanced_non_mitosis/'

numreps = 5
batch_size = 64
lambda_m = 0.01
epoch_size = 4000
Tp_min, Fp_max = 400, 100
