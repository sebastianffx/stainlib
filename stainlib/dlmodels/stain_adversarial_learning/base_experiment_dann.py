import sys
import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True
configtf.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=configtf))

import numpy as np
import pandas as pd
import os

sys.path.insert(0, 'utils')
sys.path.insert(0,'models_code')
import utils_patches
import evaluation_utils
import config4



from models_code.models_keras import dann_mitosis_model
from utils_patches import simplePatchGenerator, simplePatchGeneratorMitosis, simplePatchGeneratorDomains
from utils_patches import color_augment_patches
from evaluation_utils import evaluate_model_validation, evaluate_model_test

import keras.backend as K
from keras_utils import LR_SGD, reset_weights

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

training_gen = simplePatchGenerator(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/training/', 
    batch_size=64,
    img_shape = (63,63,3),
    augmentation_fn=None,
    return_slide_num = True,
    only_domain = False,
    balance_batch=True
)

LR_mult_dict = {'dom_regressor':0.25, "mit_pred":1} #This allows to have a lr of 0.0025 for the domain classifier: 0.0025 = 0.01 * x

sgd_opt = LR_SGD(lr=config4.lambda_m, momentum=0.9, decay=0.0005, nesterov=False, multipliers = LR_mult_dict)

lmit = K.variable(1.) # To be able to modify them later.
ldom = K.variable(1.)
K.set_value(lmit,1.)
K.set_value(ldom,0.)

losses = {
    "mit_pred":      "categorical_crossentropy",
    "dom_regressor": "categorical_crossentropy" } #categorical_crossentropy vs categorical_crossentropy

history_losses = []
model_weights_filepath = '/home/sebastian/stain_adversarial_learning/models/'+config4.TRAINING_MODE+'/' 
history_filepath       = model_weights_filepath +'history_lambdaFixed_' 

if not os.path.exists(model_weights_filepath):
    os.makedirs(model_weights_filepath)


repetition_meassures = []
change_iter = 1000
for repetition in range(config4.numreps):
    max_f1 ,best_val_thres, best_auc = 0,0,0
    #For each repetitions the model weights must be reinitialized
    model_dann = dann_mitosis_model(nTrainSlideNums = 8, hp_lambda_domain = K.variable(1))
    model_dann.compile(loss=losses,
              loss_weights=[lmit,ldom],
      optimizer=sgd_opt,
      metrics=['acc'])
    for iter_num in range(40000): #number of batches to warmup the mitosis model 
        batch_images,mitosis_labels = training_gen_mitosis.next()    
        #mitosis features gets updated
        K.set_value(lmit,1.)
        K.set_value(ldom,0.)
        K.set_value(gg.hp_lambda,0.)
        batch_images,mitosis_labels = training_gen_mitosis.next()
        history_losses.append(model_dann.train_on_batch(x=batch_images,y=[mitosis_labels, np.zeros((batch_size,8))]))
        print(history_losses[-1],str(iter_num))
        #the domain features gets updated
        K.set_value(lmit,0.)
        K.set_value(ldom,1.)
        K.set_value(gg.hp_lambda,-1.)
        batch_images,domain_labels = training_gen_domain.next()
        history_losses.append(model_dann.train_on_batch(x=batch_images,y=domain_labels))
        print(history_losses[-1],str(iter_num))
        #Adversarial update
        K.set_value(lmit,1.)
        #alpha_val = (2 /(1+np.exp(-20.*iter_num/change_iter))) - 1.
        K.set_value(ldom,1)
        K.set_value(gg.hp_lambda,1.)
        batch_images,batch_labels = training_gen.next()
        history_losses.append(model_dann.train_on_batch(x=batch_images,y=batch_labels))        
        print(history_losses[-1],str(iter_num))

        if iter_num%change_iter == 0:            
            df = pd.DataFrame(history_losses, columns=model_dann.metrics_names)
            cur_val_thres,cur_max_f1,roc_auc = evaluate_model_validation(model_dann,
                                                                '/mnt/nas3/bigdatasets/Desuto/tupac_mitosis/patches/hnm_set/val/'
                                                                 ,perf_meassure='f1',func_model=True)
            if  cur_max_f1 > max_f1:
                best_val_thres = cur_val_thres
                best_auc = roc_auc
                max_f1  = cur_max_f1
                full_path_model= model_weights_filepath+config4.TRAINING_MODE+'_weights_'+str(config4.lambda_m)+'changeiter_'+str(change_iter) + '_iter_'+str(iter_num)+'_rep_'+str(repetition)+'.hdf5'
                model_dann.save_weights(full_path_model)
                #print("Saving model for iter " + str(iter_num) +' - F1: ' +str(max_f1) + full_path_model)
           #reset_weights(model_dann,layers_to_reset) #to avoid getting stuck in local maxima and to ensure that domain information is not recovered over iterations in the main branch
    #loading the weights of the best model in validation to report test performance    
    model_dann.load_weights(full_path_model)
    roc_auc_rep_int, f1_test_rep_int = evaluate_model_test(model_dann,config4.PATH_TEST_INTERNAL_POS,config4.PATH_TEST_INTERNAL_NEG,best_val_thres)
    roc_auc_rep_ext, f1_test_rep_ext = evaluate_model_test(model_dann,config4.PATH_TEST_EXTERNAL_POS,config4.PATH_TEST_EXTERNAL_NEG,best_val_thres)
    repetition_meassures.append([roc_auc_rep_int, f1_test_rep_int,roc_auc_rep_ext, f1_test_rep_ext])

#Saving average performance meassures over test dataset
df.to_csv(history_filepath + str(config4.lambda_m) + '_iter_'+str(iter_num)+'.log', index=False) 
aggregated_meassures = [np.mean(repetition_meassures,axis=0),np.std(repetition_meassures,axis=0)]
np.savetxt(model_weights_filepath+config4.TRAINING_MODE+'_measures_dann_nowarmup_fixd_lambda.csv', repetition_meassures, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
np.savetxt(model_weights_filepath+config4.TRAINING_MODE+'_AGG_measures_dann_nowarmup_fixd_lambda.csv', aggregated_meassures, fmt='%.6e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)