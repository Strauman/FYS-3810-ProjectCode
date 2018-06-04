import os
from sys import exit
from keras.models import Sequential, Model
#------ Activations ------#
from keras.layers import LeakyReLU, Activation
#------ NN layers ------#
from keras.layers import Conv2DTranspose, Conv2D, Dense, UpSampling2D, UpSampling3D
#------ Regularizers and others ------#
from keras.layers import BatchNormalization, Flatten, Reshape, Input, concatenate, Lambda, Dropout
from keras.regularizers import l2
#------ Optimizers ------#
from keras.optimizers import Adam
# import tf.layers.conv2d_transpose
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer
import generators
from model import random_dim, image_size
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
K.set_image_dim_ordering('tf')

def dense_clean():
    return Sequential([
        Dense(7*7*128, input_shape=(random_dim+10,)),
        LeakyReLU(0.2),
        Dense(28*28),
        Activation("sigmoid"),
        Reshape((28,28,1)),
    ])
