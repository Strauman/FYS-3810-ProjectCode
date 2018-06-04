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
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
K.set_image_dim_ordering('tf')

# Dimention of the random input vector
random_dim = 10
# Images are squared with 28x28:
image_size=28

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def actual_accuracy(y_true, y_pred):
    thr_predicted_values=K.round(y_pred)
    meanwrong=K.mean(K.abs(y_true-thr_predicted_values))
    return 1-meanwrong

metrics=[actual_accuracy, mean_pred]

def model_discriminator():
    dropout_prob=0.3
    return Sequential([
        Conv2D(32, (5,5), strides=2, padding='same', input_shape=(image_size,image_size,11)),
        Dropout(dropout_prob),
        LeakyReLU(0.2),
        Conv2D(64, (5,5), strides=2, padding='same'),
        LeakyReLU(0.2),
        Conv2D(256, (5,5), strides=2, padding='same'),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1),
        Activation('sigmoid')
    ])

def build_model(generator_architecture):
    # Process the mnist_labels:
    # Reshape to fit discriminator
    mnist_labels_input=Input(shape=(10,))
    mnist_labels_discriminator=Reshape((1,1,10))(mnist_labels_input)
    mnist_labels_discriminator=Lambda(lambda y: K.ones((image_size,image_size,10))*y)(mnist_labels_discriminator)
    # Get images and concatenate with mnist_labels_discriminator
    image_inputs=Input(shape=(image_size,image_size,1,))
    discriminator_input=concatenate([image_inputs,mnist_labels_discriminator], axis=3)
    # Load the discriminator architecture, define inputs and compile
    discriminator_architecture=model_discriminator()
    discriminator=Model(inputs=[image_inputs,mnist_labels_input], outputs=discriminator_architecture(discriminator_input))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=metrics)
    ## Set up the adversarial model
    ## Do not train discriminator when training D(G(Z))
    discriminator.trainable=False
    random_input=Input(shape=(random_dim,))
    generator_input=concatenate([random_input,mnist_labels_input], axis=1)
    generator_architecture=generator_architecture()
    generator=Model(inputs=[random_input,mnist_labels_input], outputs=generator_architecture(generator_input))

    Z_input=Input(shape=(random_dim,), name="Z_input")
    L_input=Input(shape=(10,), name="L_input")
    XHat=generator([Z_input,L_input])
    GD=discriminator([XHat, L_input])
    adversarial=Model(inputs=[Z_input, L_input], outputs=GD)
    adversarial.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=metrics)
    return generator, adversarial, discriminator
