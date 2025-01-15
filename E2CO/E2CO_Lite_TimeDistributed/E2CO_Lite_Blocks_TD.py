from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Conv3D, TimeDistributed
from keras.models import Model
from keras.layers import Add, Concatenate, PReLU
import math

tf.compat.v1.enable_eager_execution() 
##########################################
#----------ENCODER NETWORK------------#  
##########################################
def create_encoder(input_shape, latent_dim):
    """
    Creates the encoder part of the Variational Autoencoder (VAE) with TimeDistributed applied 
    to ensure each realization is processed independently, but using the same network parameters.
    """
    # Input shape: (None, 5, 60, 60, 4, 2) 
    encoder_input = Input(shape=input_shape, name='encoder_input')  # (batch_size, 5, 60, 60, 4, 2)
    
    # Apply TimeDistributed to Conv3D layers to process each realization independently
    x = conv_bn_relu_multiple(input_shape, nb_filter = 16, nb_row = 3, nb_col = 3, nb_depth = 3, stride=(2, 2, 2))(encoder_input)
    #x = conv_bn_relu_multiple_Layer(num_realz=input_shape[0], num_filter=16, num_row=3, num_col=3, num_depth=3, stride=(2, 2, 2))(encoder_input)
    x = TimeDistributed(conv_bn_relu_2(num_filter=32, num_row=3, num_col=3, num_depth=3, stride=(2, 2, 2)))(x)  # (batch_size, 5, 15, 15, 1, 32)
    x = TimeDistributed(conv_bn_relu_2(num_filter=64, num_row=3, num_col=3, num_depth=3, stride=(2, 2, 2)))(x)  # (batch_size, 5, 8, 8, 1, 64)
   
    # Apply TimeDistributed to the dense block to process realizations independently but with shared weights
    x = dense_block(num_layers=3, num_filter=32, kernel_size_full=(3, 3, 3), 
                                    pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                                    padding='same', type="original")(x)  # (batch_size, 5, 8, 8, 1, 160)

    # Apply TimeDistributed to Conv3D
    Et = TimeDistributed(Conv3D(filters=192, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same'))(x)  # (batch_size, 5, 8, 8, 1, 192)
    
    # Apply TimeDistributed to Flatten each realization independently
    x = TimeDistributed(Flatten())(Et)  # (batch_size, 5, flattened_size)
    
    # Apply TimeDistributed to Dense layer to map to latent dimension independently for each realization
    encoder_output = TimeDistributed(Dense(latent_dim), name='encoder_output')(x)  # (batch_size, 5, latent_dim)

    # Create the encoder model
    encoder = Model(encoder_input, [encoder_output, Et], name='encoder')
    
    return encoder

##########################################
#----------DECODER NETWORK------------#  
##########################################

def create_decoder(input_shape, latent_dim):
    """
    Decoder network with TimeDistributed applied to process multiple realizations independently
    """
    # Decoder input for latent variable (batch_size, 5, latent_dim)
    decoder_input = Input(shape=(input_shape[0], latent_dim), name='decoder_input')  # (batch_size, 5, latent_dim)
    
    # Skip input (batch_size, 5, Nx/8, Ny/8, Nz/8, 192)
    skip_input = Input(shape=(input_shape[0], math.ceil(input_shape[1]/8), math.ceil(input_shape[2]/8), 
                              math.ceil(input_shape[3]/8), 192), name='skip_input')
    
    ##---------Dense and reshape region
    dense_units = math.ceil(input_shape[1]/8) * math.ceil(input_shape[2]/8) * math.ceil(input_shape[3]/8) * 192
    
    # Apply TimeDistributed to Dense and Reshape
    x = TimeDistributed(Dense(dense_units))(decoder_input)
    x = TimeDistributed(Reshape((math.ceil(input_shape[1]/8), math.ceil(input_shape[2]/8), 
                                 math.ceil(input_shape[3]/8), 192)))(x)

    ##------------Merge skip connection using Concatenate
    x = TimeDistributed(Concatenate())([x, skip_input])  # (batch_size, 5, Nx/8, Ny/8, Nz/8, 384)

    ##------------Apply dense block (TimeDistributed)
    x = dense_block(num_layers=3, num_filter=32, kernel_size_full=(3, 3, 3), 
                                    pointwise_kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', 
                                    type="original")(x)

    ##------------Three transpose convolutions with TimeDistributed
    x = TimeDistributed(dconv_bn_nolinear_2(64, 3, 3, 3, stride=(2, 2, 2)))(x)  # (batch_size, 5, Nx/4, Ny/4, Nz/4, 64)
    x = TimeDistributed(dconv_bn_nolinear_2(32, 3, 3, 3, stride=(2, 2, 2)))(x)  # (batch_size, 5, Nx/2, Ny/2, Nz/2, 32)
    x = TimeDistributed(dconv_bn_nolinear_2(16, 3, 3, 3, stride=(2, 2, 2)))(x)  # (batch_size, 5, Nx, Ny, Nz, 16)

    ##------------Resize to original dimensions
    x = resize_to_original_dimension(input_shape)(x)  # Resize for each realization

    ##------------Conv3D layer for decoder output
    decoder_output = TimeDistributed(Conv3D(filters=input_shape[4], kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', activation=None, name='decoder_output'))(x)
    #decoder_output = conv_multiple(input_shape, nb_filter=input_shape[4], nb_row = 3, nb_col = 3, nb_depth = 3, stride=(1, 1, 1))(x)
    # Create decoder model
    decoder = Model([decoder_input, skip_input], decoder_output, name='decoder')
    
    return decoder




##########################################
#----------TRANSITION NETWORK------------#  
##########################################
def create_trans_encoder(input_shape, input_dim):
    '''
    Creates FC transition model.

    ''' 
    zt = Input(shape=(input_shape[0], input_dim))

    # Embed z to hz
    hidden_dim = 200
    hz = TimeDistributed(fc_bn_relu_2(hidden_dim))(zt)
    hz = TimeDistributed(fc_bn_relu_2(hidden_dim))(hz)
    hz = TimeDistributed(fc_bn_relu_2(input_dim-1))(hz) #because we concatenated dt with zt in create_transition, we need to subtract to get back to latent dimension

    trans_encoder = Model(zt, hz)

    return trans_encoder


def create_transition(input_shape, latent_dim, u_dim):
    
    """ TRANSITION """
    zt = Input(shape=(input_shape[0], latent_dim), name='transition_input')
    ut1 = Input(shape=(input_shape[0],u_dim), name='transtion_input_ut1')
    dt = Input(shape=(input_shape[0],1), name='transition_input_dt')

    zt_expand = TimeDistributed(Concatenate(axis = -1))([zt, dt])
    trans_encoder = create_trans_encoder(input_shape, latent_dim + 1)
    hz = trans_encoder(zt_expand) 

    At = TimeDistributed(Dense(latent_dim*latent_dim))(hz)
    At = TimeDistributed(Reshape((latent_dim, latent_dim)))(At)    
    Bt = TimeDistributed(Dense(latent_dim*u_dim))(hz)
    Bt = TimeDistributed(Reshape((latent_dim, u_dim)))(Bt)  

    batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    scalar_multi = Lambda(lambda x: x[0] * x[1]) # Larry Jin
    
    ut_dt = TimeDistributed(scalar_multi)([ut1, dt]) # Larry Jin
    
    term1 = TimeDistributed(batch_dot_layer)([At, zt])
    term2 = TimeDistributed(batch_dot_layer)([Bt, ut_dt])
    transition_output = TimeDistributed(Add(name='transition_output'))([term1, term2])


    transition = Model([zt, ut1, dt], transition_output, name='transition')
      
    return transition



##########################################
 #--------TRANSITION OUTPUT NETWORK-----#  
##########################################


def create_transition_out(input_shape, latent_dim, u_dim, out_dim):

    """ TRANSITION OUTPUT """
    zt = Input(shape=(input_shape[0], latent_dim), name='transition_input')
    zt1 = Input(shape=(input_shape[0], latent_dim), name='transition_out_input')
    ut1 = Input(shape=(input_shape[0], u_dim), name='transition_input_ut1')
    dt = Input(shape=(input_shape[0],1), name='transition_input_dt')


    zt_expand = TimeDistributed(Concatenate(axis=-1))([zt, dt])
    trans_out_encoder = create_trans_encoder(input_shape,latent_dim + 1)
    hz = trans_out_encoder(zt_expand)
    
    Ct = TimeDistributed(Dense(latent_dim*out_dim))(hz)
    Ct = TimeDistributed(Reshape((out_dim, latent_dim)))(Ct)         
    #Ct = TimeDistributed(Dropout(0.5))(Ct)     # Adding dropout after reshaping Ct
    Dt = TimeDistributed(Dense(u_dim*out_dim))(hz)
    Dt = TimeDistributed(Reshape((out_dim, u_dim)))(Dt)            
    #Dt = TimeDistributed(Dropout(0.5))(Dt)  # Adding dropout after reshaping Dt

    batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
    scalar_multi = Lambda(lambda x: x[0] * x[1]) # Larry Jin
    
    ut_dt = TimeDistributed(scalar_multi)([ut1, dt]) # Larry Jin
    term1 = TimeDistributed(batch_dot_layer)([Ct, zt1])
    term2 = TimeDistributed(batch_dot_layer)([Dt, ut_dt])
    
    transition_out_output = TimeDistributed(Add(name='transition_well_outputs_output'))([term1, term2])

    transition_out_output  = TimeDistributed(PReLU())(transition_out_output)  ##added relu to remove negative values

    transition_out = Model([zt, zt1, ut1, dt], [transition_out_output], name='transition_well_output')
    
    return transition_out



########################################################################
######                  E2CO LITE MODEL                            ####
#######################################################################

""" E2CO_Lite MODEL """
#======================================================================================================================
class e2co_lite(Model):
    def __init__(self, input_shape, latent_dim, u_dim, out_dim, discard_decoder = False):
        super(e2co_lite, self).__init__()
        self._shape = input_shape
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.out_dim = out_dim
        self.discard_decoder = discard_decoder
        self._build_model(self._shape, self.latent_dim, self.u_dim, self.out_dim)
    
    def _build_model(self, input_shape, latent_dim, u_dim, out_dim):
        if not self.discard_decoder:
            self.encoder = create_encoder(input_shape, latent_dim) 
            self.decoder = create_decoder(input_shape, latent_dim)
            self.transition = create_transition(input_shape, latent_dim, u_dim)
            self.transition_output = create_transition_out(input_shape, latent_dim, u_dim, out_dim)  ##E2CO transition output
        elif self.discard_decoder:
            self.encoder = create_encoder(input_shape, latent_dim)
            self.transition = create_transition(input_shape, latent_dim, u_dim)
            self.transition_output = create_transition_out(input_shape, latent_dim, u_dim, out_dim)  ##E2CO transition output

    
    def get_config(self):
        return {"input_shape" : self._shape,
                "latent_dim": self.latent_dim,
                "u_dim": self.u_dim,
                "out_dim": self.out_dim,
                "discard_decoder": self.discard_decoder} 
    
    def call(self, inputs):
        self.xt, self.ut1, self.dt = inputs
        if not self.discard_decoder:
            self.zt, self.Et = self.encoder(self.xt)
            self.xt_rec = self.decoder([self.zt, self.Et])
            self.zt1_pred = self.transition([self.zt, self.ut1, self.dt])
            self.xt1_pred = self.decoder([self.zt1_pred, self.Et])
            self.yt1_pred = self.transition_output([self.zt, self.zt1_pred, self.ut1, self.dt]) ##E2CO transition output
            output = (self.xt_rec, self.zt1_pred, self.xt1_pred, self.yt1_pred)
        elif self.discard_decoder:
            self.zt, self.Et = self.encoder(self.xt)
            self.zt1_pred = self.transition([self.zt, self.ut1, self.dt])
            self.yt1_pred = self.transition_output([self.zt, self.zt1_pred, self.ut1, self.dt]) ##E2CO transition output
            output = (self.zt1_pred, self.yt1_pred)
        
        return output
    
    def reconstruction(self, xt):
        zt, Et = self.encoder(xt)
        xt_rec = self.decoder([zt, Et])

        return xt_rec


    def use_encoder(self, xt, return_numpy=True):
        zt, Et = self.encoder(xt)
        if tf.executing_eagerly():
            if return_numpy:
                return zt.numpy(), Et.numpy()
            else:
                return zt, Et
        else:
            return zt, Et
        
    def prediction(self, inputs):
        
        zt, Et, ut1, dt = inputs
        zt1_pred = self.transition([zt, ut1, dt])
        #yt1_pred = self.transition_output([zt, zt1_pred.numpy(), ut1, dt])  #E2CO Transition out   #requires all input be of same type. so I converted zt1_pred to a numpy array
        yt1_pred = self.transition_output([zt, zt1_pred, ut1, dt])
        xt1_pred = self.decoder([zt1_pred, Et])

        return zt1_pred, xt1_pred, yt1_pred

    def prediction_state(self, inputs):
        
        zt, Et, ut1, dt = inputs      
        zt1_pred = self.transition([zt, ut1, dt])
        xt1_pred = self.decoder([zt1_pred, Et])

        return zt1_pred, xt1_pred

    def prediction_well(self, inputs, return_numpy = False):    
        zt, ut1, dt = inputs     
        zt1_pred = self.transition([zt, ut1, dt])
        yt1_pred = self.transition_output([zt, zt1_pred, ut1, dt])  #E2CO Transition out  ##requires all input be of same type. so I converted zt1_pred to a numpy array
        if tf.executing_eagerly():
            print("Executed Eagerly")
            if return_numpy:
                print("Returning Numpy array")
                return zt1_pred.numpy(), yt1_pred.numpy()
            else:
                return zt1_pred, yt1_pred
        else:
            print("Executed Symbolically")
            return zt1_pred, yt1_pred

    def loadweights(self, encoder_file, decoder_file, transition_file, transition_out_file):
        if not self.discard_decoder:
            self.encoder.load_weights(encoder_file)
            self.decoder.load_weights(decoder_file)
            self.transition.load_weights(transition_file)
            self.transition_output.load_weights(transition_out_file)
        elif self.discard_decoder:
            self.encoder.load_weights(encoder_file)
            self.transition.load_weights(transition_file)
            self.transition_output.load_weights(transition_out_file)


    def saveweights(self, encoder_file, decoder_file, transition_file, transition_out_file):
        if not self.discard_decoder:
            self.encoder.save_weights(encoder_file)
            self.decoder.save_weights(decoder_file)
            self.transition.save_weights(transition_file)
            self.transition_output.save_weights(transition_out_file)
        if self.discard_decoder:
            self.encoder.save_weights(encoder_file)
            self.transition.save_weights(transition_file)
            self.transition_output.save_weights(transition_out_file)
