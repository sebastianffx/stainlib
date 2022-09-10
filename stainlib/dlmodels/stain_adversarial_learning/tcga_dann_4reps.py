import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
configtf = tf.ConfigProto()
configtf.gpu_options.allow_growth = True
configtf.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=configtf))
import glob
import os
import sys
from PIL import * 
sys.path.insert(0, '/home/sebastian/stain_adversarial_learning/utils/')
from utils_patches import color_augment_patches

from config import *
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/sebastian/stain_adversarial_learning/models_code/')
from models_code.models_keras import GradientReversal
from keras import Model
from keras.layers.core import Dense 
from sklearn.metrics import *
from keras.applications.mobilenet import MobileNet
import keras.backend as K
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from utils.evaluation_utils import evaluate_model_test_tma,evaluate_model_validation_TCGA
from keras_utils import LR_SGD
from imp import reload
import utils.utils_patches 
reload(utils.utils_patches)
from utils.utils_patches import patchgen_tcga_only_domains, simplePatchGeneratorTCGA

#Defining batch generators
training_gen_tcga =simplePatchGeneratorTCGA(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/training/', 
    batch_size=64,
    img_shape = (224,224,3),
    augmentation_fn=None)

validation_gen_tcga =simplePatchGeneratorTCGA(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/training/', 
    batch_size=64,
    img_shape = (224,224,3),
    augmentation_fn=None)

domains_gen_tcga = patchgen_tcga_only_domains(
    input_dir='/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/all_patches/', 
    batch_size=64,
    img_shape = (224,224,3),
    augmentation_fn=None)

mpath = '/home/sebastian/stain_adversarial_learning/models/tcga_dann_gen/'

if not os.path.exists(mpath):
    os.makedirs(mpath)
LL =training_gen_tcga.next()
print(LL[0].shape)
lmit  = K.variable(1.) # To be able to modify them later.
ldom = K.variable(1.)
history_losses = []

###################Model Training 
mpath   = '/home/sebastian/stain_adversarial_learning/models/tcga_dann_4reps/'

if not os.path.exists(mpath):
    os.makedirs(mpath)

for rep in range(4):
    history_losses = []
    history_filepath    = mpath +'history_'+'rep_'+str(rep)+'_' 

    #Base model extraction from pre-trained mobilnet with alpha=0.5
    base_mnet = MobileNet(input_shape=(224,224,3), alpha=0.5, depth_multiplier=1, dropout=.2, include_top=False,
                     weights="imagenet", input_tensor=None, pooling='avg', classes=2)


    n_domains = len(training_gen_tcga.domains_dict_train) #The domain branch of the network depends on the total #domains

    hp_lambda_domain = K.variable(1)

    x_top = base_mnet.output
    gr = GradientReversal(hp_lambda=hp_lambda_domain)
    xd_reversal = gr(x_top)

    feats1_gp = Dense(256, name='feats1_gp', activation='relu')(x_top)
    feats2_gp = Dense(128, name='feats2_gp', activation='relu')(feats1_gp)
    y_gleason = Dense(2, name='output', activation='softmax')(feats2_gp)

    feats1_dom = Dense(256, name='feats1_dom', activation='relu')(xd_reversal)
    feats2_dom = Dense(128, name='feats2_dom', activation='relu')(feats1_dom)
    y_dom = Dense(n_domains, name='output_dom', activation='softmax')(feats2_dom)

    model = Model(base_mnet.input, [y_gleason,y_dom])
    #model.summary()
    losses = {
        "output":      "categorical_crossentropy",
        "output_dom": "categorical_crossentropy" } #categorical_crossentropy vs categorical_crossentropy

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional layers
    for layer in base_mnet.layers:
        layer.trainable = False

        

    LR_mult_dict = {'output':0.25, "output_dom":1} #This allows to have a lr of 0.0025 for the domain classifier: 0.0025 = 0.01 * x

    sgd_opt = LR_SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False, multipliers = LR_mult_dict)

    #With the SGD LR multipliers the modifiers don't work deterministically 
    model.compile(optimizer=optimizers.Adam(lr=0.001), 
                      loss=losses, loss_weights=[lmit,ldom], metrics = ['accuracy'])
    lgp  = K.variable(1.) # To be able to modify them later.
    ldom = K.variable(1.)

    print(model.metrics_names)

    K.set_value(model.loss_weights[0],1.)
    K.set_value(model.loss_weights[1],0.)

    gg = model.layers[-7]
    K.set_value(gg.hp_lambda,0.)

    history = model.fit_generator(
                generator=training_gen_tcga,
                steps_per_epoch=20,
                epochs=1,
                validation_data=training_gen_tcga,
                validation_steps=2,
                verbose=1
    )


    change_iter = 100
    #for iter_num in range(4000):  
    print(model.metrics_names)
    for epoch in range(5):
        for iter_num in range(1000):
            #GP features gets updated
            K.set_value(lmit,1.)
            K.set_value(ldom,0.)
            K.set_value(model.loss_weights[0],1.)
            K.set_value(model.loss_weights[1],0.)
            K.set_value(gg.hp_lambda,0.)
            batch_images,labels = training_gen_tcga.next()
            history_losses.append(model.train_on_batch(x=batch_images,y=labels))
            print(history_losses[-1],str(iter_num))

            #the domain features gets updated
            K.set_value(lmit,1.)
            K.set_value(ldom,1.)
            K.set_value(model.loss_weights[0],0.)
            K.set_value(model.loss_weights[1],1.)
            K.set_value(gg.hp_lambda,-1.)
            batch_images,domain_labels = domains_gen_tcga.next()
            history_losses.append(model.train_on_batch(x=batch_images,y=domain_labels))
            print(history_losses[-1],str(iter_num))

            #Adversarial update
            K.set_value(lmit,1.)
            #alpha_val = (2 /(1+np.exp(-20.*iter_num/change_iter))) - 1.
            K.set_value(ldom,1)
            K.set_value(model.loss_weights[0],1.)
            K.set_value(model.loss_weights[1],1.)
            K.set_value(gg.hp_lambda,1.)
            batch_images,labels = training_gen_tcga.next()
            history_losses.append(model.train_on_batch(x=batch_images,y=labels))
            print(history_losses[-1],str(iter_num))
        ###
        df = pd.DataFrame(history_losses, columns=model.metrics_names)
        full_path_model_weights = mpath + 'weights_'+ str(lambda_m)+'_rep_'+str(rep) + '_epoch_' + str(epoch)  +'_iter_' + str(iter_num)+'.hdf5'
        model.save_weights(full_path_model_weights)
    df.to_csv(history_filepath + str(lambda_m) + '_iter_'+str(iter_num)+'.log', index=False)

all_external_gp3 = glob.glob('/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/external_test/GP3/*jpg')
all_external_gp4 = glob.glob('/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/external_test/GP4/*jpg')
all_internal_gp3 = glob.glob('/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/internal_test/GP3/*png')
all_internal_gp4 = glob.glob('/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/internal_test/GP4/*png')

validation_images_path = '/mnt/nas3/bigdatasets/Desuto/tcga/graded_diagnostic_patch_dataset/subset_stain_norm/val/'
val_models_paths = []
models_measures  = []
for rep in range(4):
    model_weights_paths = glob.glob(mpath + 'weights_'+ str(lambda_m)+'_rep_'+str(rep)+'*hdf5')
    best_val_path, best_val_thres,best_val_max_f1,best_val_roc_auc = '',0,0,0
    for wpath in model_weights_paths:
        model.load_weights(wpath)
        best_thres,max_f1,roc_auc = evaluate_model_validation_TCGA(model,validation_images_path,perf_meassure='f1', func_model=True)
        if roc_auc > best_val_roc_auc:
            best_val_roc_auc = roc_auc
            best_val_max_f1 = max_f1
            best_val_thres = best_thres
            best_val_path = wpath
            print("Better validation model found with the weights path: " + wpath)
            print(best_val_roc_auc,best_val_max_f1,best_val_thres)
    val_models_paths.append(best_val_path)
    model.load_weights(best_val_path)
    int_auc, int_f1 = evaluate_model_test_tma(model,all_internal_gp3,all_internal_gp4,best_thres,
                        return_probs=False, func_model=True,internal_tcga_test = True)
    ext_auc, ext_f1 = evaluate_model_test_tma(model,all_external_gp3,all_external_gp4,best_thres,
                        return_probs=False, func_model=True)
    models_measures.append([int_auc, int_f1,ext_auc,ext_f1])
print(val_models_paths)
print('int_auc,    int_f1,    ext_auc,   ext_f1')
print(np.mean(models_measures,axis=0))
print(np.std(models_measures,axis=0))
