import os
import glob
import numpy as np
import tensorflow as tf
from skimage.io import imsave
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, 
    UpSampling2D, Add, Multiply, Lambda, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Define Atrous Spatial Pyramid Pooling (ASPP) block for feature extraction at multiple scales
def aspp_block(x, num_filters, rate_scale=1):
    # Create dilation rates for multiple scaled features
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
    x2 = BatchNormalization()(x2)
    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
    x3 = BatchNormalization()(x3)
    x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
    x4 = BatchNormalization()(x4)
    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="same")(y)
    return y

# Function to expand a tensor's size to match another
def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)

# Attention gating block to focus on specific features
def AttnGatingBlock(x, g, inter_shape):
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), 
                                 strides=(K.int_shape(theta_x)[1] // K.int_shape(phi_g)[1], K.int_shape(theta_x)[2] // K.int_shape(phi_g)[2]),
                                 padding='same')(phi_g)
    concat_xg = Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(K.int_shape(x)[1] // K.int_shape(sigmoid_xg)[1], K.int_shape(x)[2] // K.int_shape(sigmoid_xg)[2]))(sigmoid_xg)
    upsample_psi = expend_as(upsample_psi, K.int_shape(x)[3])
    y = Multiply()([upsample_psi, x])
    result = Conv2D(K.int_shape(x)[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def UnetGatingSignal(inputs, is_batchnorm=False):
    shape = K.int_shape(inputs)
    x = Conv2D(shape[3] * 2, (1, 1), padding="same")(inputs)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Define custom Conv1x1 Layer
class Conv1x1(tf.keras.layers.Layer):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = tf.keras.layers.Conv2D(planes, kernel_size=1)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Sobel operator functions for boundary enhancement
def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = tf.sqrt(tf.pow(g_x, 2) + tf.pow(g_y, 2))
    return tf.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    # Sobel filters for edge detection
    filter_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=tf.float32)
    filter_y = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)
    filter_x = tf.reshape(filter_x, (1, 1, 3, 3))
    filter_y = tf.reshape(filter_y, (1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)
    filter_y = np.repeat(filter_y, out_chan, axis=0)
    filter_x = filter_x.transpose(2, 3, 0, 1)
    filter_y = filter_y.transpose(2, 3, 0, 1)
    filter_x = tf.convert_to_tensor(filter_x, dtype=tf.float32)
    filter_y = tf.convert_to_tensor(filter_y, dtype=tf.float32)
    filter_x = tf.Variable(filter_x, trainable=True)
    filter_y = tf.Variable(filter_y, trainable=True)
    conv_x = tf.keras.layers.Conv2D(out_chan, kernel_size=3, strides=1, padding='same', use_bias=False)
    conv_y = tf.keras.layers.Conv2D(out_chan, kernel_size=3, strides=1, padding='same', use_bias=False)
    conv_x.build((None, None, None, in_chan))
    conv_y.build((None, None, None, in_chan))
    conv_x.set_weights([filter_x.numpy()])
    conv_y.set_weights([filter_y.numpy()])
    sobel_x = tf.keras.Sequential([conv_x, tf.keras.layers.BatchNormalization()])
    sobel_y = tf.keras.Sequential([conv_y, tf.keras.layers.BatchNormalization()])
    return sobel_x, sobel_y

# Define the full model with UNet architecture enhanced by attention gates and boundary enhancement

def seg_model():
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = aspp_block(pool4, 512) 
    gating = UnetGatingSignal(conv5, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 256)
    sobel_x, sobel_y = get_sobel(attn_1.shape[-1], attn_1.shape[-1])  # Adjust input and output channels accordingly
    BEM1 = run_sobel(sobel_x, sobel_y, attn_1)
    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv5), BEM1], axis=3)  
    
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    gating = UnetGatingSignal(conv6, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 128)
    sobel_x, sobel_y = get_sobel(attn_2.shape[-1], attn_2.shape[-1])  # Adjust input and output channels accordingly
    BEM2 = run_sobel(sobel_x, sobel_y, attn_2)
    up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv6), BEM2], axis=3) 
    
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    gating = UnetGatingSignal(conv7, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 64)
    sobel_x, sobel_y = get_sobel(attn_3.shape[-1], attn_3.shape[-1])  # Adjust input and output channels accordingly
    BEM3 = run_sobel(sobel_x, sobel_y, attn_3)

    up8 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv7), BEM3], axis=3) 
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)   
    
    gating = UnetGatingSignal(conv8, is_batchnorm=True)
    attn_4 = AttnGatingBlock(conv1, gating, 32)
    sobel_x, sobel_y = get_sobel(attn_4.shape[-1], attn_4.shape[-1])  # Adjust input and output channels accordingly
    BEM4 = run_sobel(sobel_x, sobel_y, attn_4)

    up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv8), BEM4], axis=3) 
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv9 = aspp_block(conv9, 32)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    return model
