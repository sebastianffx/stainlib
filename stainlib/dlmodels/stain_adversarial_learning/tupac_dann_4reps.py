import glob
import sys
sys.path.insert(0, '/home/sebastian/local_experiments/staining/utils/')
#import config
#from config import *
import numpy as np

import sys
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True
configtf.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=configtf))
import glob
import numpy as np
import pandas as pd
import os
sys.path.insert(0, 'utils')
sys.path.insert(0,'models_code')
from models_code.models_keras import dann_mitosis_model
from utils_patches import simplePatchGeneratorMitosis, simplePatchGeneratorDomains,patchgen_tupac_only_domains
from evaluation_utils import evaluate_model_validation, evaluate_model_test

import keras.backend as K
from keras_utils import LR_SGD

TRAINING_MODE = 'dann_unsup_tupac' 
PATH_GT = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/mitoses_ground_truth/'

PATH_TEST_INTERNAL_POS = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/test/mitosis/'
PATH_TEST_INTERNAL_NEG = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/test/non_mitosis/'

PATH_TEST_EXTERNAL_POS = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/external/mitosis/'
PATH_TEST_EXTERNAL_NEG = '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/external/balanced_non_mitosis/'

numreps    = 4
batch_size = 64
lambda_m   = 0.01


training_gen_mitosis = simplePatchGeneratorMitosis(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/training/', 
    batch_size=64,
    img_shape = (63,63,3),
    augmentation_fn=None,
    balance_batch=True
)

training_gen_domain = simplePatchGeneratorDomains(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/training/', 
    batch_size=64,
    img_shape = (63,63,3),
    augmentation_fn=None
)

training_gen_domain = patchgen_tupac_only_domains(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/all_patches/', 
    batch_size=64,
    img_shape = (63,63,3),
    augmentation_fn=None)



LR_mult_dict = {'dom_regressor':0.25, "mit_pred":1} #This allows to have a lr of 0.0025 for the domain classifier: 0.0025 = 0.01 * x

sgd_opt = LR_SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False, multipliers = LR_mult_dict)

lmit = K.variable(1.) # To be able to modify them later.
ldom = K.variable(1.)
K.set_value(lmit,1.)
K.set_value(ldom,0.)

losses = {
    "mit_pred":      "categorical_crossentropy",
    "dom_regressor": "categorical_crossentropy" } #categorical_crossentropy vs categorical_crossentropy

model_weights_filepath = '/home/sebastian/stain_adversarial_learning/models/tupac_dann_rep/' 
history_filepath       = model_weights_filepath +'history_' 

if not os.path.exists(model_weights_filepath):
    os.makedirs(model_weights_filepath)

repetition_meassures = []

change_iter = 1000
max_f1 ,best_val_thres, best_auc, model_paths = 0,0,0, []

path_patches = glob.glob('/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/all_patches/*png')


for rep in range(4): #4 repetitions of the same training scheme
    print("=================TRAINING REPETITION " + str(rep)+" FOR MODEL "+model_weights_filepath +"==========")
    history_losses = []
    model_dann = dann_mitosis_model(nTrainSlideNums = len(training_gen_domain.domains_dict_train),
                                    hp_lambda_domain = K.variable(1))

    model_dann.compile(loss=losses,
              loss_weights=[lmit,ldom],
      optimizer=sgd_opt,
      metrics=['acc'])
    gg = model_dann.layers[-7]

    #Mitosis branch warmup
    print(model_dann.metrics_names)
    for iter_num in range(20): #number of batches to warmup the mitosis model 
        K.set_value(model_dann.loss_weights[0],1.)
        K.set_value(model_dann.loss_weights[1],0.)
        K.set_value(lmit,1.)
        K.set_value(ldom,0.)
        K.set_value(gg.hp_lambda,0.)
        batch_images,mitosis_labels = training_gen_mitosis.next()
        history_losses.append(model_dann.train_on_batch(x=batch_images,y=mitosis_labels))
        print(history_losses[-1],str(iter_num))

    for epoch in range(5): #now the iterative training of the DANN model begins
        for iter_num in range(1000): 
            #Mitosis branch update
            K.set_value(model_dann.loss_weights[0],1.)
            K.set_value(model_dann.loss_weights[1],0.)
            K.set_value(lmit,1.)
            K.set_value(ldom,0.)
            K.set_value(gg.hp_lambda,0.)
            batch_images,mitosis_labels = training_gen_mitosis.next()
            history_losses.append(model_dann.train_on_batch(x=batch_images,y=mitosis_labels))
            print(history_losses[-1],str(iter_num))
            #Domain branch update
            K.set_value(model_dann.loss_weights[0],0.)
            K.set_value(model_dann.loss_weights[1],1.)
            K.set_value(lmit,0.)
            K.set_value(ldom,1.)
            K.set_value(gg.hp_lambda,-1.)
            batch_images,mitosis_labels = training_gen_domain.next()
            history_losses.append(model_dann.train_on_batch(x=batch_images,y=mitosis_labels))
            print(history_losses[-1],str(iter_num))
            #Adversarial update
            K.set_value(lmit,1.)
            #alpha_val = (2 /(1+np.exp(-20.*iter_num/change_iter))) - 1.
            K.set_value(ldom,1)
            K.set_value(model_dann.loss_weights[0],1.)
            K.set_value(model_dann.loss_weights[1],1.)
            K.set_value(gg.hp_lambda,1.)
            batch_images,mitosis_labels = training_gen_mitosis.next()
            history_losses.append(model_dann.train_on_batch(x=batch_images,y=mitosis_labels))
            print(history_losses[-1],str(iter_num))

        cur_val_thres,cur_max_f1,roc_auc = evaluate_model_validation(model_dann,
                                                            '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/val/'
                                                            ,perf_meassure='f1',func_model=True)
        if  cur_max_f1 > max_f1:
            best_val_thres = cur_val_thres
            best_auc = roc_auc
            max_f1  = cur_max_f1
            full_path_model= model_weights_filepath+'weights_'+str(lambda_m)+'epoch_'+str(epoch) + '_iter_'+str(iter_num)+'_rep_'+str(rep)+'.hdf5'
            model_dann.save_weights(full_path_model)
    #loading the weights of the best model in validation to report test performance    
    model_dann.load_weights(full_path_model)
    model_paths.append(full_path_model)
    roc_auc_rep_int, f1_test_rep_int = evaluate_model_test(model_dann,PATH_TEST_INTERNAL_POS,PATH_TEST_INTERNAL_NEG,best_val_thres)
    roc_auc_rep_ext, f1_test_rep_ext = evaluate_model_test(model_dann,PATH_TEST_EXTERNAL_POS,PATH_TEST_EXTERNAL_NEG,best_val_thres)
    repetition_meassures.append([roc_auc_rep_int, f1_test_rep_int,roc_auc_rep_ext, f1_test_rep_ext])
    df = pd.DataFrame(history_losses, columns=model_dann.metrics_names)

df.to_csv(history_filepath + str(lambda_m) + '_iter_'+str(iter_num)+'.log', index=False)
aggregated_meassures = [np.mean(repetition_meassures,axis=0),np.std(repetition_meassures,axis=0)]
np.savetxt(model_weights_filepath+TRAINING_MODE+'_measures_dann_lambda.csv', repetition_meassures, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt(model_weights_filepath+TRAINING_MODE+'_AGG_measures_dann.csv', aggregated_meassures, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
print(model_paths)
print(aggregated_meassures)