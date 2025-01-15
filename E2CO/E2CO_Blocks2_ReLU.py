from E2CO.E2CO_Layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, BatchNormalization
from keras.models import Model
from keras.layers import Add, Multiply, Concatenate


# Function 1: ENCODER Network:
def create_encoder(input_shape, latent_dim):
    """
    ENCODER

    :param input_shape: dimension of the input (up to 3D tensor)
    :param latent_dim: dimension of the encoder output (latent dimension)
    :return: encoder

    """
    encoder_input = Input(shape=input_shape, name='encoder_input')

    # x = conv_bn_relu(nb_filter=16, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(encoder_input)
    # x = conv_bn_relu(nb_filter=32, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    # x = conv_bn_relu(nb_filter=64, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)
    # x = conv_bn_relu(nb_filter=128, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    #
    x = conv_bn_relu_multiple(input_shape, nb_filter=16, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(encoder_input)
    x = conv_bn_relu_multiple(input_shape, nb_filter=32, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    x = conv_bn_relu_multiple(input_shape, nb_filter=64, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)
    x = conv_bn_relu_multiple(input_shape, nb_filter=128, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)

    for i in range(3):
        # x = res_conv(nb_filter=128, nb_row=3, nb_col=3, nb_depth=3)(x)
        x = res_conv_multiple(input_shape, nb_filter=128, nb_row=3, nb_col=3, nb_depth=3)(x)

    NumberOfRealizations = input_shape[0]
    flatten_out = []
    for Realization in range(NumberOfRealizations):
        out = Flatten()(x[:, Realization, ...])
        dkm = Dense(latent_dim, name='encoder_output_' + str(Realization + 1))(out)
        flatten_out.append(tf.expand_dims(dkm, axis=1))
    encoder_output = tf.concat(flatten_out, axis=1)

    encoder = Model(encoder_input, encoder_output, name='encoder')

    return encoder

# Function 2: Transition Block inside TRANSITION and TRANSITION OUTPUT networks:
def create_trans_encoder(input_dim, hidden_dim=200):
    """


    :param input_dim: input dimension (1D)
    :param hidden_dim: number of hidden units in between
    :return: transition encoder
    """

    zt = Input(shape=(input_dim, ))

    hz = fc_bn_relu(hidden_dim)(zt)
    hz = fc_bn_relu(hidden_dim)(hz)
    hz = fc_bn_relu(input_dim - 1)(hz)

    trans_encoder = Model(zt, hz)

    return trans_encoder


# Function 3: TRANSITION Network:
def create_transition(input_shape, latent_dim, controls_dim):
    """
    TRANSITION

    :param latent_dim: input latent dimension (as output of the encoder)
    :param controls_dim: dimension of the well control u_t
    :return: transition
    """

    zt = Input(shape=(input_shape[0], latent_dim), name='transition_input')
    ut1 = Input(shape=controls_dim, name='transtion_input_ut1')
    dt = Input(shape=1, name='transition_input_dt')

    transition_output_final = []
    for Realization in range(input_shape[0]):
        zt_expand = Concatenate(axis=-1)([zt[:, Realization, ...], dt])
        trans_encoder = create_trans_encoder(input_dim=(latent_dim + 1), hidden_dim=200)
        hz = trans_encoder(zt_expand)

        At = Dense(latent_dim*latent_dim)(hz)
        At = Reshape((latent_dim, latent_dim))(At)
        Bt = Dense(latent_dim * controls_dim)(hz)
        Bt = Reshape((latent_dim, controls_dim))(Bt)

        batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
        scalar_multiply = Lambda(lambda x: x[0] * x[1])

        ut_dt = scalar_multiply([ut1, dt])

        term1 = batch_dot_layer([At, zt[:, Realization, ...]])
        term2 = batch_dot_layer([Bt, ut_dt])
        transition_output = Add(name='transition_output_' + str(Realization + 1))([term1, term2])
        transition_output_final.append(tf.expand_dims(transition_output, axis=1))
    transition_output_final = tf.concat(transition_output_final, axis=1)

    transition = Model([zt, ut1, dt], [transition_output_final], name='transition')

    return transition


# Function 4: TRANSITION OUTPUT Network:
def create_transition_out(input_shape, latent_dim, controls_dim, output_dim):

    zt = Input(shape=(input_shape[0], latent_dim), name='transition_input')
    zt1 = Input(shape=(input_shape[0], latent_dim), name='transtion_out_input')
    ut1 = Input(shape=controls_dim, name='transtion_input_ut1')
    dt = Input(shape=1, name='transition_input_dt')

    transition_out_output_final = []
    for Realization in range(input_shape[0]):
        zt_expand = Concatenate(axis=-1)([zt[:, Realization, ...], dt])
        trans_out_encoder = create_trans_encoder(input_dim=(latent_dim + 1), hidden_dim=200)
        hz = trans_out_encoder(zt_expand)

        Ct = Dense(latent_dim * output_dim)(hz)
        Ct = Reshape((output_dim, latent_dim))(Ct)
        Dt = Dense(controls_dim * output_dim)(hz)
        Dt = Reshape((output_dim, controls_dim))(Dt)

        batch_dot_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]))
        scalar_multiply = Lambda(lambda x: x[0] * x[1])

        ut_dt = scalar_multiply([ut1, dt])

        term1 = batch_dot_layer([Ct, zt1[:, Realization, ...]])
        term2 = batch_dot_layer([Dt, ut_dt])

        transition_out_output = Add(name='transition_out_output_' + str(Realization + 1))([term1, term2])

        transition_out_output_final.append(tf.expand_dims(transition_out_output, axis=1))
    transition_out_output_final = tf.concat(transition_out_output_final, axis=1)
    transition_out_output_final = tf.keras.layers.ReLU()(transition_out_output_final)  # Newly added to prevent negative well outputs

    transition_out = Model([zt, zt1, ut1, dt], [transition_out_output_final], name='transition_out')

    return transition_out


# Function 5: DECODER Network:
def create_decoder(input_shape, latent_dim):
    """

    :param input_shape:
    :param latent_dim:
    :return:
    """

    decoder_input = Input(shape=(input_shape[0], latent_dim), name='decoder_input')

    dkm = []
    for Realization in range(input_shape[0]):
        x = Dense(int((input_shape[1]/4) * (input_shape[2]/4) * (int(input_shape[3]/4) + (input_shape[3] % 4 > 0)) * 128), activation='relu')(decoder_input[:, Realization, ...])
        x = Reshape((int(input_shape[1]/4), int(input_shape[2]/4), int(input_shape[3]/4) + (input_shape[3] % 4 > 0), 128))(x)
        dkm.append(tf.expand_dims(x, axis=1))
    x = tf.concat(dkm, axis=1)

    for i in range(3):
        # x = res_conv(nb_filter=128, nb_row=3, nb_col=3, nb_depth=3)(x)
        x = res_conv_multiple(input_shape, nb_filter=128, nb_row=3, nb_col=3, nb_depth=3)(x)

    # x = dconv_bn_nolinear(nb_filter=128, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    # x = dconv_bn_nolinear(nb_filter=64, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)
    # x = dconv_bn_nolinear(nb_filter=32, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    # x = dconv_bn_nolinear(nb_filter=16, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)

    x = dconv_bn_nolinear_multiple(input_shape, nb_filter=128, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    x = dconv_bn_nolinear_multiple(input_shape, nb_filter=64, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)
    x = dconv_bn_nolinear_multiple(input_shape, nb_filter=32, nb_row=3, nb_col=3, nb_depth=3, stride=(1, 1, 1))(x)
    x = dconv_bn_nolinear_multiple(input_shape, nb_filter=16, nb_row=3, nb_col=3, nb_depth=3, stride=(2, 2, 2))(x)

    dkm = []
    for Realization in range(input_shape[0]):
        out = Conv3D(input_shape[4], (3, 3, 3), padding='same', activation="relu", name='decoder_output_' + str(Realization + 1))(x[:, Realization, ...])
        dkm.append(tf.expand_dims(out, axis=1))

    decoder_output = tf.concat(dkm, axis=1)
    del dkm, out

    decoder = Model(decoder_input, decoder_output, name='decoder')

    return decoder



class e2co(Model):
    def __init__(self, input_shape, latent_dim, controls_dim, output_dim):
        super(e2co, self).__init__()
        self._build_model(input_shape, latent_dim, controls_dim, output_dim)

    def _build_model(self, input_shape, latent_dim, controls_dim, output_dim):
        self.encoder = create_encoder(input_shape=input_shape, latent_dim=latent_dim)
        # print(self.encoder.summary())
        self.transition = create_transition(input_shape=input_shape, latent_dim=latent_dim, controls_dim=controls_dim)
        # print(self.transition.summary())
        self.transition_output = create_transition_out(input_shape=input_shape, latent_dim=latent_dim, controls_dim=controls_dim, output_dim=output_dim)
        # print(self.transition_output.summary())
        self.decoder = create_decoder(input_shape=input_shape, latent_dim=latent_dim)
        # print(self.decoder.summary())


    def call(self, inputs):
        self.xt, self.ut1, self.dt = inputs

        # Pass xt to the encoder to get zt:
        self.zt = self.encoder(self.xt)

        # Reconstruct xt by passing zt to the decoder:
        self.xt_reconstruct = self.decoder(self.zt)

        # Do the same for predictions of z_(t+1) and x_(t+1):
        self.zt1_predict = self.transition([self.zt, self.ut1, self.dt])
        self.xt1_predict = self.decoder(self.zt1_predict)

        # Prediction for well output y_(t+1):
        self.yt1_predict = self.transition_output([self.zt, self.zt1_predict, self.ut1, self.dt])

        return self.xt_reconstruct, self.zt1_predict, self.xt1_predict, self.yt1_predict

    def reconstruction(self, xt):
        xt_rec = self.decoder(self.encoder(xt))
        return xt_rec

    def prediction(self, inputs, predict_well_output=True):
        zt, ut1, dt = inputs

        zt1_predict = self.transition([zt, ut1, dt])
        xt1_predict = self.decoder(zt1_predict)
        if predict_well_output:
            yt1_predict = self.transition_output([zt, zt1_predict, ut1, dt])
            return zt1_predict, xt1_predict, yt1_predict
        else:
            return zt1_predict, xt1_predict

    def prediction_well_output_only(self, inputs):
        zt, ut1, dt = inputs

        zt1_predict = self.transition([zt, ut1, dt])
        yt1_predict = self.transition_output([zt, zt1_predict, ut1, dt])

        return zt1_predict, yt1_predict

    def loadweights(self, encoder_file, decoder_file, transition_file, transition_out_file):
        self.encoder.load_weights(encoder_file)
        self.decoder.load_weights(decoder_file)
        self.transition.load_weights(transition_file)
        self.transition_output.load_weights(transition_out_file)


    def saveweights(self, encoder_file, decoder_file, transition_file, transition_out_file):
        self.encoder.save_weights(encoder_file)
        self.decoder.save_weights(decoder_file)
        self.transition.save_weights(transition_file)
        self.transition_output.save_weights(transition_out_file)

