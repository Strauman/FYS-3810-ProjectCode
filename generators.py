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
# random_dim and image_size is available when imported
# into model
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
K.set_image_dim_ordering('tf')
random_dim=10
def dense_clean():
    return Sequential([
        Dense(7*7*128, input_shape=(random_dim+10,)),
        LeakyReLU(0.2),
        Dense(28*28),
        Activation("sigmoid"),
        Reshape((28,28,1)),
    ])
def dense_dropout():
    dropout_prob=0.3
    return Sequential([
        Dense(7*7*128, input_shape=(random_dim+10,)),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(28*28),
        Activation("sigmoid"),
        Reshape((28,28,1)),
    ])
def dense_batchnorm():
    dropout_prob=0.3
    lambda2=0
    return Sequential([
        Dense(7*7*128, input_shape=(random_dim+10,)),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(28*28),
        Activation("sigmoid"),
        Reshape((28,28,1)),
    ])
def dense_batchnorm_dropout():
    dropout_prob=0.3
    lambda2=0
    return Sequential([
        Dense(7*7*128, input_shape=(random_dim+10,)),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dropout(0.3),
        Dense(28*28),
        Activation("sigmoid"),
        Reshape((28,28,1)),
    ])
def conv_clean():
    return Sequential([
        Reshape((1,1,random_dim+10), input_shape=(random_dim+10,)),
        UpSampling2D(14),
        Conv2D(128, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        UpSampling2D(),
        Conv2D(64, 7, padding='same'),
        LeakyReLU(0.2),
        UpSampling2D(),
        Conv2D(1, 8, padding='same'),
        Activation("sigmoid"),
    ])
def conv_dropout():
    return Sequential([
        Reshape((1,1,random_dim+10), input_shape=(random_dim+10,)),
        UpSampling2D(14),
        Conv2D(128, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        UpSampling2D(),
        Conv2D(64, 7, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.3),
        UpSampling2D(),
        Conv2D(1, 8, padding='same'),
        Activation("sigmoid"),
    ])
def conv_batchnorm():
    return Sequential([
        Reshape((1,1,random_dim+10), input_shape=(random_dim+10,)),
        UpSampling2D(14),
        Conv2D(128, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(64, 7, padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(1, 8, padding='same'),
        Activation("sigmoid"),
    ])
def conv_batchnorm_dropout():
    dropout_prob=0.3
    return Sequential([
        Reshape((1,1,random_dim+10), input_shape=(random_dim+10,)),
        UpSampling2D(14),
        Conv2D(128, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dropout(dropout_prob),
        UpSampling2D(),
        Conv2D(64, 7, padding='same'),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dropout(dropout_prob),
        UpSampling2D(),
        Conv2D(1, 8, padding='same'),
        Activation("sigmoid"),
    ])
