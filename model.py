import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import os
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50

def resblock(x, f):
    # function for creating res block

    x_copy = x

    x = Conv2D(f, kernel_size=1, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f, kernel_size=3, padding='same', dilation_rate=7, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    x_copy = Conv2D(f, kernel_size=1, kernel_initializer='he_normal')(x_copy)
    x_copy = BatchNormalization()(x_copy)

    x = Add()([x, x_copy])
    x = Activation('relu')(x)

    return x

def upsample_concat(x, skip, filters):
    x = Conv2DTranspose(filters, kernel_size=2, strides=(2, 2), padding='same')(x)
    
    merge = Concatenate()([x, skip])

    return merge

def resnet_encoder(x_input):
    # stage 1
    conv_1 = resblock(x_input, 16)
    pool_1 = MaxPooling2D((2, 2))(conv_1)

    # stage 2
    conv_2 = resblock(pool_1, 32)
    pool_2 = MaxPooling2D((2, 2))(conv_2)

    # stage 3
    conv_3 = resblock(pool_2, 64)
    pool_3 = MaxPooling2D((2, 2))(conv_3)

    # stage 4
    conv_4 = resblock(pool_3, 128)
    pool_4 = MaxPooling2D((2, 2))(conv_4)

    # stage 5
    conv_5 = resblock(pool_4, 192)
    pool_5 = MaxPooling2D((2, 2))(conv_5)

    return conv_1, conv_2, conv_3, conv_4, conv_5, pool_5

def resnet_decoder(conv_6, conv_5, conv_4, conv_3, conv_2, conv_1):
    # Upsample Stage 1
    up_1 = upsample_concat(conv_6, conv_5, 192)
    up_1 = resblock(up_1, 192)

    # Upsample Stage 2
    up_2 = upsample_concat(up_1, conv_4, 128)
    up_2 = resblock(up_2, 128)

    # Upsample Stage 3
    up_3 = upsample_concat(up_2, conv_3, 64)
    up_3 = resblock(up_3, 64)

    # Upsample Stage 4
    up_4 = upsample_concat(up_3, conv_2, 32)
    up_4 = resblock(up_4, 32)

    # Upsample Stage 5
    up_5 = upsample_concat(up_4, conv_1, 16)
    up_5 = resblock(up_5, 16)

    return up_5

def get_resnet_model(input_shape, n_classes):
    input_1 = Input(shape=input_shape, name="input_1")
    input_2 = Input(shape=input_shape, name="input_2")
    input_3 = Input(shape=input_shape, name="input_3")

    conv_1_input_1, conv_2_input_1, conv_3_input_1, conv_4_input_1, conv_5_input_1, pool_5_input_1 = resnet_encoder(input_1)

    conv_1_input_2, conv_2_input_2, conv_3_input_2, conv_4_input_2, conv_5_input_2, pool_5_input_2 = resnet_encoder(input_2)

    conv_1_input_3, conv_2_input_3, conv_3_input_3, conv_4_input_3, conv_5_input_3, pool_5_input_3 = resnet_encoder(input_3)

    # stage 6 (bottle neck)
    input_concat = Concatenate(axis=-1)([pool_5_input_1, pool_5_input_2, pool_5_input_3])
    conv_6 = resblock(input_concat, 128)

    up_input_1 = resnet_decoder(conv_6, conv_5_input_1, conv_4_input_1, conv_3_input_1, conv_2_input_1, conv_1_input_1)
    up_input_2 = resnet_decoder(conv_6, conv_5_input_2, conv_4_input_2, conv_3_input_2, conv_2_input_2, conv_1_input_2)
    up_input_3 = resnet_decoder(conv_6, conv_5_input_3, conv_4_input_3, conv_3_input_3, conv_2_input_3, conv_1_input_3)

    up_concat = Concatenate(axis=-1)([up_input_1, up_input_2, up_input_3])

    # Output layer
    out = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', padding='same', activation='softmax')(up_concat)

    model = Model([input_1, input_2, input_3], out)

    '''
    for layer in model.layers:
        print(layer.name, layer.trainable)
    '''


    return model

def get_single_input_resnet_model(input_shape, n_classes):
    input_1 = Input(shape=input_shape, name="input_1")

    conv_1_input_1, conv_2_input_1, conv_3_input_1, conv_4_input_1, conv_5_input_1, pool_5_input_1 = resnet_encoder(input_1)

    # stage 6 (bottle neck)
    conv_6 = resblock(pool_5_input_1, 128)

    up_input_1 = resnet_decoder(conv_6, conv_5_input_1, conv_4_input_1, conv_3_input_1, conv_2_input_1, conv_1_input_1)

    # Output layer
    out = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal', padding='same', activation='softmax')(up_input_1)

    model = Model(input_1, out)
    model.summary()

    '''
    for layer in model.layers:
        print(layer.name, layer.trainable)
    '''

    return model

if __name__ == '__main__':
    model = get_resnet_model((224, 224, 1), 16)
    model.summary()