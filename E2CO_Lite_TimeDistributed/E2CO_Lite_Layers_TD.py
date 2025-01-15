import tensorflow as tf

from keras.layers import Layer, Dense, BatchNormalization, Activation, Conv3D, Cropping3D, Add
from keras.layers import PReLU, Concatenate, DepthwiseConv2D, UpSampling3D, Lambda, ReLU, Conv3DTranspose, TimeDistributed
from keras.models import Sequential
import keras
tf.compat.v1.enable_eager_execution() 


###################################################
####        CONVOLUTION  AND FC BLOCK          ####
###################################################

##Convolution block
def conv_bn_relu(num_filter, num_row, num_col, num_depth, stride):
    
    def conv_func(x):
        x = Conv3D(num_filter, (num_row, num_col, num_depth), strides=stride, padding='same')(x)
        x = BatchNormalization(fused=False)(x)
        x = ReLU()(x) 
        return x

    return conv_func

def conv(num_filter, num_row, num_col, num_depth, stride, padding, activation, name):
    
    def convolution(x):
        x = Conv3D(num_filter, (num_row, num_col, num_depth), strides=stride, padding=padding, activation=activation, name=name)(x)
        return x

    return convolution

def conv_bn_relu_2(num_filter, num_row, num_col, num_depth, stride):
    model = Sequential([
        Conv3D(num_filter, (num_row, num_col, num_depth), strides=stride, padding='same'),
        BatchNormalization(fused=False),
        ReLU()
    ])
    return model

## Multimodel Convolution
def conv_bn_relu_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2)):
    def conv_func_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = conv_bn_relu(nb_filter, nb_row, nb_col, nb_depth, stride)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return conv_func_multiple

def conv_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2)):
    def conv_func_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = conv(nb_filter, nb_row, nb_col, nb_depth, stride, padding='same', activation=None, name='decoder_output' + '_Realization' + str(Realization))(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return conv_func_multiple

##Fully Connected (FC) block with relu
def fc_bn_relu(hidden_dim):
    
    def fc_func(x):
        x = Dense(hidden_dim, activation=None)(x)
        x = BatchNormalization(fused=False)(x)
        x = ReLU()(x)
        return x

    return fc_func

def fc_bn_relu_2(hidden_dim):
    
    model = Sequential([
         Dense(hidden_dim, activation=None),
         BatchNormalization(fused=False),
         ReLU()])
    return model

def res_conv(nb_filter, nb_row, nb_col, nb_depth, stride=(1, 1, 1), activation='relu'):
    def _res_func(x):
        identity = x

        a = TimeDistributed(Conv3D(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same'))(x)
        a = TimeDistributed(BatchNormalization())(a)
        a = TimeDistributed(Activation(activation))(a)
        a = TimeDistributed(Conv3D(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same'))(a)
        y = TimeDistributed(BatchNormalization())(a)

        return TimeDistributed(Add())([identity, y])

    return _res_func

###################################################
####        DECONVOLUTION ROUTINES             ####
###################################################


# Deconvolution using transpose convolution 
def dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2)):
    
    def _dconv_bn(x):
        x = Conv3DTranspose(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same')(x)
        x = BatchNormalization(fused=False)(x)
        x = ReLU()(x)
        return x

    return _dconv_bn

def dconv_bn_nolinear_2(nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2)):
    
    model = Sequential([
         Conv3DTranspose(nb_filter, (nb_row, nb_col, nb_depth), strides=stride, padding='same'),
         BatchNormalization(fused=False),
         ReLU()
        ])
    
    return model

    return _dconv_bn
#Multimodel Deconvolution using transpose convolution
def dconv_bn_nolinear_multiple(input_shape, nb_filter, nb_row, nb_col, nb_depth, stride=(2, 2, 2)):
    def dconv_bn_nonlinear_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = dconv_bn_nolinear(nb_filter, nb_row, nb_col, nb_depth, stride)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return dconv_bn_nonlinear_multiple

# Deconvolution using DWS and Upsampling
def deconv_block(pointwise_filters, depthwise_kernel_size=(3, 3, 3), pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same'):
    
    def DWS(x):
         #Pointwise convolution with 1X1X1 kernels
        x = Conv3D(filters=pointwise_filters, kernel_size=pointwise_kernel_size, strides=(1, 1, 1), padding=padding)(x)
        x = BatchNormalization(fused=False)(x)
        x = ReLU()(x)
        
        #Upsampling by 2
        x = UpSampling3D(size = (2,2,2))(x)  #upsample X2
        
        #DW
        x = DepthwiseConv3D(depthwise_kernel_size=depthwise_kernel_size, strides=strides, padding=padding)(x)
        
        x = BatchNormalization(fused=False)(x)

        x = ReLU()(x)
        
        return x  
    
    return DWS

# Multimodel Deconvolution using DWS and Upsampling
def deconv_block_multiple(input_shape, pointwise_filters, depthwise_kernel_size=(3, 3, 3), pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same'):
    def _deconv_block_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = deconv_block(pointwise_filters, depthwise_kernel_size, pointwise_kernel_size, strides, padding)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return _deconv_block_multiple  


## At the time this was implemented, DepthwiseConv3D has not been implemented in tensorflow.    
class DepthwiseConv3D(Layer):
    def __init__(self, depthwise_kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', **kwargs):
        super(DepthwiseConv3D, self).__init__(**kwargs)
        self.depthwise_kernel_size = depthwise_kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.depth_slices = input_shape[-2]
        self.depthwise_convs = [
            DepthwiseConv2D(kernel_size=self.depthwise_kernel_size[:-1], strides=self.strides[:-1], padding=self.padding)
            for _ in range(self.depth_slices)
        ]

    def call(self, inputs):
        # Split the input tensor into slices along the depth axis (last axis before the channel axis)
        slices = tf.split(inputs, self.depth_slices, axis=-2)
        
        # Apply DepthwiseConv2D to each slice
        depthwise_slices = []
        for i, slice in enumerate(slices):
            # Remove the singleton depth dimension to convert (batch, height, width, 1, channels) -> (batch, height, width, channels)
            slice_4d = tf.squeeze(slice, axis=-2)
            # Apply DepthWiseConv2D on each 2D slice
            depthwise_slice = self.depthwise_convs[i](slice_4d)
            
            # Add the singleton depth dimension back to convert (batch, height, width, channels) -> (batch, height, width, 1, channels)
            depthwise_slice = tf.expand_dims(depthwise_slice, axis=-2)
            depthwise_slices.append(depthwise_slice)
        
        # Concatenate the depthwise slices back along the depth axis
        return tf.concat(depthwise_slices, axis=-2)
    



############################################################################
###       DENSE-NET ROUTINES  - A REPLACEMENT FOR RESIDUAL NETS         ####
############################################################################


## Dense block --> A slight variant of DenseNet
def dense_block(num_layers, num_filter, kernel_size_full = (3, 3, 3), pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', type = "modified"):
    
    def dense_block_func(x):
        for _ in range(num_layers):
            x = conv_bn_relu_for_dense_net(x, growth_rate = num_filter, kernel_size_full=kernel_size_full, pointwise_kernel_size=pointwise_kernel_size, strides=strides, padding=padding, type = type)
        return x
    
    return dense_block_func


## Multimodel denseblock
def dense_block_multiple(input_shape, num_layers, num_filter, kernel_size_full = (3, 3, 3), pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', type = "modified"):
    def _dense_block_multiple(x):
        out = []
        NumberOfRealizations = input_shape[0]
        for Realization in range(NumberOfRealizations):
            dkm = dense_block(num_layers, num_filter, kernel_size_full, pointwise_kernel_size, strides, padding, type)(x[:, Realization, ...])
            out.append(tf.expand_dims(dkm, axis=1))
        return tf.concat(out, axis=1)
    return _dense_block_multiple


def conv_bn_relu_for_dense_net(x, growth_rate, kernel_size_full=(3, 3, 3), pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', type = "modified"):
    if type.lower() == "original": ##original Implementation from DenseNet paper
        x1 = TimeDistributed(BatchNormalization(fused=False))(x)
        x1 = TimeDistributed(ReLU())(x1)
        x1 = TimeDistributed(Conv3D(filters=4*growth_rate, kernel_size=pointwise_kernel_size, strides=(1, 1, 1), padding=padding))(x1) #1X1X1 pointwise bottleneck
        x1 = TimeDistributed(BatchNormalization(fused=False))(x1)
        x1 = TimeDistributed(ReLU())(x1)
        x1 = TimeDistributed(Conv3D(filters=growth_rate, kernel_size = kernel_size_full, strides=strides, padding=padding))(x1)
        x = TimeDistributed(Concatenate())([x, x1])
    elif type.lower() == "modified": ##implementation from Hu Huang et. al.
        x1 = TimeDistributed(Conv3D(filters=4*growth_rate, kernel_size=pointwise_kernel_size, strides=(1, 1, 1), padding='same'))(x) #1X1X1 pointwise bottleneck
        x1 = TimeDistributed(BatchNormalization(fused=False))(x1)
        x1 = TimeDistributed(ReLU())(x1)
        x1 = TimeDistributed(Conv3D(filters=growth_rate, kernel_size = kernel_size_full, strides=strides, padding='same'))(x1)
        x1 = TimeDistributed(BatchNormalization(fused=False))(x1)
        x1 = TimeDistributed(ReLU())(x1)
        x = TimeDistributed(Concatenate())([x, x1])
    return x


###########################################################
###      RESIZING TO ORIGINAL INPUT DIMENSIONS         ####
###########################################################

def resize_to_original_dimension_old_version(input_shape):
    
    def resize(x):
        # If the input dimensions are larger, use cropping3D
        if x.shape[2] > input_shape[1] or x.shape[3] > input_shape[2] or x.shape[4] > input_shape[3]:
            x1 = x.shape[1]
            x2 = x.shape[2]
            x3 = x.shape[3]
            crop_x1 = (x.shape[2] - input_shape[1]) // 2
            crop_y1 = (x.shape[3] - input_shape[2]) // 2
            crop_z1 = (x.shape[4] - input_shape[3]) // 2

            crop_x2 = (x.shape[2] - input_shape[1]) - crop_x1
            crop_y2 = (x.shape[3] - input_shape[2]) - crop_y1
            crop_z2 = (x.shape[4] - input_shape[3]) - crop_z1

            x = TimeDistributed(Cropping3D(cropping=((crop_x1, crop_x2), (crop_y1, crop_y2), (crop_z1, crop_z2))))(x)         
        return x
    return resize


# E2CO_Lite_Layers_TD.py
def resize_to_original_dimension(input_shape):
    
    def resize(x):
        # If the input dimensions are larger, use Cropping3D
        if x.shape[2] > input_shape[1] or x.shape[3] > input_shape[2] or x.shape[4] > input_shape[3]:
            # Convert TensorFlow Dimension objects to integers
            x1 = int(x.shape[1])
            x2 = int(x.shape[2])
            x3 = int(x.shape[3])
            crop_x1 = int((x2 - input_shape[1]) // 2)
            crop_y1 = int((x3 - input_shape[2]) // 2)
            crop_z1 = int((x.shape[4] - input_shape[3]) // 2)

            crop_x2 = int((x2 - input_shape[1]) - crop_x1)
            crop_y2 = int((x3 - input_shape[2]) - crop_y1)
            crop_z2 = int((x.shape[4] - input_shape[3]) - crop_z1)

            # Define cropping as integers
            cropping = (
                (crop_x1, crop_x2),
                (crop_y1, crop_y2),
                (crop_z1, crop_z2),
            )

            # Apply Cropping3D with integer cropping values
            x = TimeDistributed(Cropping3D(cropping=cropping))(x)         
        return x
    return resize

#@keras.saving.register_keras_serializable(package="MyLayers")
#@tf.keras.utils.register_keras_serializable()

class conv_bn_relu_multiple_Layer(tf.keras.layers.Layer):
    def __init__(self, num_realz = 10, num_filter=16, num_row=3, num_col=3, num_depth=3, stride=(2, 2, 2), **kwargs):
        super(conv_bn_relu_multiple_Layer, self).__init__(**kwargs)
        # Store parameters
        self.num_realz = num_realz
        self.num_filter = num_filter
        self.num_row = num_row
        self.num_col = num_col
        self.num_depth = num_depth
        self.stride = stride
        self.conv_layers = [conv_bn_relu_2(self.num_filter, self.num_row, self.num_col, self.num_depth, self.stride) for _ in range(self.num_realz)]

    
    def call(self, inputs):
        """
        Process each realization in the input tensor.
        Args:
        - inputs: Tensor of shape [batch_size, NumberOfRealizations, D, H, W, C].
        
        Returns:
        - Tensor of shape [batch_size, NumberOfRealizations, new_D, new_H, new_W, num_filter].
        """
        # Get the number of realizations
        number_of_realizations = inputs.shape[1]  # Assuming axis 1 corresponds to NumberOfRealizations
        
        # Collect processed realizations
        out = []
        for realization_idx in range(number_of_realizations):
            # Slice the input for the current realization
            realization = inputs[:, realization_idx, ...]  # Shape: [batch_size, D, H, W, C]
            # Apply the convolutional layer
            dkm = self.conv_layers[realization_idx](realization)
            #dkm = self.conv_layer(realization)  # Shape: [batch_size, new_D, new_H, new_W, num_filter]
            # Expand the dimension to reintroduce the realizations axis
            out.append(tf.expand_dims(dkm, axis=1))

        # Concatenate along the realizations axis
        return tf.concat(out, axis=1)  # Shape: [batch_size, NumberOfRealizations, new_D, new_H, new_W, num_filter]
    
    def get_config(self):
        return {"num_realz" : self.num_realz,
                "num_filter": self.num_filter,
                "num_row": self.num_row,
                "num_col": self.num_col,
                "num_depth": self.num_depth,
                "stride": self.stride} 
