# from tensorflow.python.keras.layers import Dense, BatchNormalization, Activation, Conv3D, Conv3DTranspose, add

import tensorflow as tf
import numpy as np
from keras.layers import Dense, BatchNormalization, Activation, Conv3D, Conv3DTranspose, add

# Function 1: Fully connected ReLU layer (Transformation Block):
def fc_bn_relu(hidden_dim):
    def fc_func(x):
        x = Dense(hidden_dim, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return fc_func

# Function 2: Convolutional Encoding block with ReLU activation by default:
def conv_bn_relu(nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2), activation="relu"):
    def conv_func(x):
        x = Conv3D(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return conv_func

# Function 2: Convolutional Encoding block with ReLU activation by default FOR MULTIPLE REALIZATIONS:
def conv_bn_relu_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2), activation='relu'):
    def conv_func_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = conv_bn_relu(nb_filter, nb_row, nb_col, nb_depth, stride, activation)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return conv_func_multiple


# Function 3: Residual block (in both Encoder and Decoder):
def res_conv(nb_filter, nb_row, nb_col, nb_depth, stride=(1, 1, 1), activation='relu'):
    def _res_func(x):
        identity = x

        a = Conv3D(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same')(x)
        a = BatchNormalization()(a)
        a = Activation(activation)(a)
        a = Conv3D(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same')(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


# Function 4: Residual block (in both Encoder and Decoder) FOR MULTIPLE REALIZATIONS:
def res_conv_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(1, 1, 1), activation="relu"):
    def res_func_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = res_conv(nb_filter, nb_row, nb_col, nb_depth, stride, activation)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return res_func_multiple


# Function 5: Transpose-convolutional Decoding block with ReLU activation by default:
def dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2), activation="relu"):
    def _dconv_bn(x):
        x = Conv3DTranspose(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn


# Function 6: Transpose-convolutional Decoding block with ReLU activation by default:
def dconv_bn_nolinear_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(1, 1, 1), activation="relu"):
    def dconv_bn_nonlinear_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_depth, stride, activation)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return dconv_bn_nonlinear_multiple