import sys
sys.path.insert(0,'../utils/')
import config
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input
from keras.layers import Dense, Conv2D,Concatenate

from keras.applications.densenet import DenseNet121
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras import activations, Model
from keras.engine import Layer
from keras.layers import Dense, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten, Dense, Dropout, Activation 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import *
from matplotlib.pyplot import imread
import random
sys.path.insert(0, 'utils')
import utils_patches
import evaluation_utils
import config
import importlib
sys.path.insert(0,'models_code')

from keras_utils import LR_SGD, reset_weights#, GradientReversal


#Definition of the CNN model
kernel_size = (4,4) #Should be this in config?
input_shape = (63,63,3)
nb_filters = 16 
pool_size = 2


def dann_mitosis_model(nTrainSlideNums=8, hp_lambda_domain = K.variable(1)):
    ##Block 1
    input_x = Input(shape=(63,63,3,),dtype="float") 
    #input_slides = Input
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), 
               kernel_initializer="glorot_uniform", activation='relu',padding='valid')(input_x)
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu',padding='valid')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    ##Block 2
    x2 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = MaxPooling2D(pool_size=pool_size)(x2)
    x2 = Dropout(0.25)(x2)

    ##Block 3
    x3 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x2)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = MaxPooling2D(pool_size=pool_size)(x3)
    x3 = Dropout(0.25)(x3)

    ##Block 4
    x4 = Flatten()(x3)
    gr = GradientReversal(hp_lambda=hp_lambda_domain)
    xd_reversal = gr(x4)
    x5 = Dense(128,activation='relu',name='dom_pred_feats1')(xd_reversal)
    x5 = Dense(128,activation='relu',name='dom_pred_feats2')(x5)

    x4 = Dense(128,activation='relu',name='mit_pred_feats1')(x4)

    #Here there are two branches: One for the cancer detection and the other for the domain
    x4 = Dense(128,activation='relu',name='mit_pred_feats2')(x4)

    ydomain = Dense(nTrainSlideNums, name='dom_regressor',activation='softmax')(x5) #domain regressor

    x4 = BatchNormalization(axis=-1)(x4)
    x4 = Activation("relu")(x4)
    x4 = Dropout(0.25)(x4)
    x4 = Dense(2,activation='softmax',name='mit_pred')(x4)

    model_d1 = Model(outputs=[x4,ydomain], inputs=input_x)
    #plot_model(model, to_file='/home/sebastian/stain_adversarial_learning/models/FL_LAYER_model.png')
    
    return model_d1



def mitosis_model(lr,clip_norm):
    model_mitosis = Sequential() #All outputs depend only of the previous layer
    #Block 0
    model_mitosis.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='valid',
                            input_shape=input_shape))
    model_mitosis.add(Activation('relu'))
    #Block 1
    model_mitosis.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),activation='relu'))
    model_mitosis.add(BatchNormalization())
    model_mitosis.add(Activation('relu'))
    model_mitosis.add(MaxPooling2D(pool_size=pool_size))
    model_mitosis.add(Dropout(0.25))
    #Block 2
    model_mitosis.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),activation='relu'))
    model_mitosis.add(BatchNormalization())
    model_mitosis.add(Activation('relu'))
    model_mitosis.add(MaxPooling2D(pool_size=pool_size))
    model_mitosis.add(Dropout(0.25))
    #Block 3
    model_mitosis.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),activation='relu'))
    model_mitosis.add(BatchNormalization())
    model_mitosis.add(Activation('relu'))
    model_mitosis.add(MaxPooling2D(pool_size=pool_size))
    model_mitosis.add(Dropout(0.25))
    model_mitosis.add(Flatten())
    #Block 4
    model_mitosis.add(Dense(128))#, activation= 'relu')
    model_mitosis.add(BatchNormalization())
    model_mitosis.add(Activation('softmax'))
    model_mitosis.add(Dropout(0.25))
    model_mitosis.add(Dense(2,activation='softmax'))

    #Defining the optimizer
    sgd_opt = SGD(lr=lr, momentum=0.9, decay=0.9, nesterov=True)

    if clip_norm:
        adam_opt = Adam(lr,clipnorm=1,decay=0.9)
    else:
        adam_opt = Adam(lr)

    model_mitosis.compile(loss='categorical_crossentropy',
              optimizer=adam_opt ,
              metrics=['mae','acc'])
    return model_mitosis



def reverse_gradient(X, hp_lambda):# where hp_lambda is the constant which multiplies the flipped gradient.
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        
        #self.hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda,name='hp_lambda')
        
    def build(self, input_shape):
        self.trainable_weights = []
        #super(GradientReversal, self).build(input_shape)

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  'hp_lambda': K.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def dann_mitosis_model_GAP(nTrainSlideNums=8,hp_lambda_domain = K.variable(1)):
    ##Block 1
    input_x = Input(shape=(63,63,3,),dtype="float") 
    #input_slides = Input
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), 
               kernel_initializer="glorot_uniform", activation='relu',padding='valid')(input_x)
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu',padding='valid')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    ##Block 2
    x2 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = MaxPooling2D(pool_size=pool_size)(x2)
    x2 = Dropout(0.25)(x2)

    ##Block 3
    x3 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x2)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = MaxPooling2D(pool_size=pool_size)(x3)
    x3 = Dropout(0.25)(x3)

    ##Block 4
    ##Block 4
    x4 = GlobalAveragePooling2D()(x3)
    #x4 = Flatten()(x3)
    gr = GradientReversal(hp_lambda=hp_lambda_domain)
    xd_reversal = gr(x4)
    x5 = Dense(128,activation='relu',name='dom_pred_feats1')(xd_reversal)
    x5 = Dense(128,activation='relu',name='dom_pred_feats2')(x5)

    x4 = Dense(128,activation='relu',name='mit_pred_feats1')(x4)

    #Here there are two branches: One for the cancer detection and the other for the domain
    x4 = Dense(128,activation='relu',name='mit_pred_feats2')(x4)

    ydomain = Dense(nTrainSlideNums, name='dom_regressor',activation='softmax')(x5) #domain regressor

    x4 = BatchNormalization(axis=-1)(x4)
    x4 = Activation("relu")(x4)
    x4 = Dropout(0.25)(x4)
    x4 = Dense(2,activation='softmax',name='mit_pred')(x4)

    model_d1 = Model(outputs=[x4,ydomain], inputs=input_x)
    model_d2 = Model(outputs=[x4,ydomain], inputs=input_x)
    #plot_model(model, to_file='/home/sebastian/stain_adversarial_learning/models/FL_LAYER_model.png')
    
    return model_d1, model_d2

def dann_mitosis_model_WODO(nTrainSlideNums=8,hp_lambda_domain = K.variable(1)):
    ##Block 1
    input_x = Input(shape=(63,63,3,),dtype="float") 
    #input_slides = Input
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), 
               kernel_initializer="glorot_uniform", activation='relu',padding='valid')(input_x)
    x = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu',padding='valid')(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=pool_size)(x)

    ##Block 2
    x2 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = MaxPooling2D(pool_size=pool_size)(x2)
    #x2 = Dropout(0.25)(x2)

    ##Block 3
    x3 = Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), kernel_initializer="glorot_uniform", activation='relu')(x2)
    x3 = BatchNormalization(axis=-1)(x3)
    x3 = MaxPooling2D(pool_size=pool_size)(x3)
    #x3 = Dropout(0.25)(x3)

    ##Block 4
    ##Block 4
    x4 = GlobalAveragePooling2D()(x3)
    #x4 = Flatten()(x3)
    gr = GradientReversal(hp_lambda=hp_lambda_domain)
    xd_reversal = gr(x4)
    x5 = Dense(128,activation='relu',name='dom_pred_feats1')(xd_reversal)
    x5 = Dense(128,activation='relu',name='dom_pred_feats2')(x5)

    x4 = Dense(128,activation='relu',name='mit_pred_feats1')(x4)

    #Here there are two branches: One for the cancer detection and the other for the domain
    x4 = Dense(128,activation='relu',name='mit_pred_feats2')(x4)

    ydomain = Dense(nTrainSlideNums, name='dom_regressor',activation='softmax')(x5) #domain regressor

    x4 = BatchNormalization(axis=-1)(x4)
    x4 = Activation("relu")(x4)
    #x4 = Dropout(0.25)(x4)
    x4 = Dense(2,activation='softmax',name='mit_pred')(x4)

    model_d1 = Model(outputs=[x4,ydomain], inputs=input_x)
    model_d2 = Model(outputs=[x4,ydomain], inputs=input_x)
    #plot_model(model, to_file='/home/sebastian/stain_adversarial_learning/models/FL_LAYER_model.png')
    
    return model_d1, model_d2
