import tensorflow as tf
# tf.keras.backend.set_floatx('float64')

from E2CO.E2CO_Blocks2_ReLU import e2co
from E2CO.E2CO_Loss2 import e2co_loss

from contextlib import redirect_stdout
import os

##This is a wrapper for the main e2co model
class e2co_model():
    def __init__(self, learning_rate, input_shape, latent_dim, u_dim, out_dim, weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well,
                 SWATTableFile=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\SWAT_table.txt',
                 GRIDMinMax=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\GridMinMax.txt',
                 NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm',
                 Nx=60, Ny=60, Nz=4):

        self.model = e2co(input_shape, latent_dim, u_dim, out_dim)
        self.loss_e2co = e2co_loss(weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well, SWATTableFile,
                 GRIDMinMax,NumberOfRealizations, PermFolder, Nx, Ny, Nz)

        self.train_total_loss = tf.keras.metrics.Mean(name='train/total_loss')
        self.train_rec_loss = tf.keras.metrics.Mean(name='train/rec_loss')
        self.train_pred_loss = tf.keras.metrics.Mean(name='train/pred_loss')
        self.train_trans_loss = tf.keras.metrics.Mean(name='train/trans_loss')
        self.train_flux_loss = tf.keras.metrics.Mean(name='train/flux_loss')
        self.train_well_loss = tf.keras.metrics.Mean(name='train/well_loss')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_rec_loss = tf.keras.metrics.Mean(name='val/rec_loss')
        self.val_pred_loss = tf.keras.metrics.Mean(name='val/pred_loss')
        self.val_trans_loss = tf.keras.metrics.Mean(name='val/trans_loss')
        self.val_flux_loss = tf.keras.metrics.Mean(name='val/flux_loss')
        self.val_well_loss = tf.keras.metrics.Mean(name='val/well_loss')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_rec_loss = tf.keras.metrics.Mean(name='test/rec_loss')
        self.test_pred_loss = tf.keras.metrics.Mean(name='test/pred_loss')
        self.test_trans_loss = tf.keras.metrics.Mean(name='test/trans_loss')
        self.test_flux_loss = tf.keras.metrics.Mean(name='test/flux_loss')
        self.test_well_loss = tf.keras.metrics.Mean(name='test/well_loss')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    def write_model_summary(self, encoder_file=r'./EncoderSummary.txt', transition_file=r'./TransitionSummary.txt',
                            transition_output_file=r'./TransitionOutputSummary.txt',
                            decoder_file=r'./DecoderSummary.txt'):
        os.makedirs(os.path.dirname(encoder_file), exist_ok=True)
        os.makedirs(os.path.dirname(transition_file), exist_ok=True)
        os.makedirs(os.path.dirname(transition_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(decoder_file), exist_ok=True)

        with open(encoder_file, 'w+') as f:
            with redirect_stdout(f):
                self.model.encoder.summary(line_length=200)

        with open(transition_file, 'w+') as f:
            with redirect_stdout(f):
                self.model.transition.summary(line_length=200)

        with open(transition_output_file, 'w+') as f:
            with redirect_stdout(f):
                self.model.transition_output.summary(line_length=200)

        with open(decoder_file, 'w+') as f:
            with redirect_stdout(f):
                self.model.decoder.summary(line_length=200)

    def reconstruct(self, xt):
        xt_rec = self.model.reconstruction(xt)
        if tf.executing_eagerly():
            return xt_rec.numpy()
        else:
            return xt_rec

    def use_encoder(self, xt, return_numpy=True):
        zt = self.model.encoder(xt)
        if tf.executing_eagerly():
            if return_numpy:
                return zt.numpy()
            else:
                return zt
        else:
            return zt

    def predict(self, inputs, return_numpy=True):
        zt1_predict, xt1_predict, yt1_predict = self.model.prediction(inputs, predict_well_output=True)
        if tf.executing_eagerly():
            if return_numpy:
                return zt1_predict.numpy(), xt1_predict.numpy(), yt1_predict.numpy()
            else:
                return zt1_predict, xt1_predict, yt1_predict
        else:
            return zt1_predict, xt1_predict, yt1_predict

    def predict_state(self, inputs, return_numpy=True):
        zt1_predict, xt1_predict = self.model.prediction(inputs, predict_well_output=False)
        if tf.executing_eagerly():
            if return_numpy:
                return zt1_predict.numpy(), xt1_predict.numpy()
            else:
                return zt1_predict, xt1_predict
        else:
            return zt1_predict, xt1_predict

    def predict_output(self, inputs, return_numpy=True):
        zt1_predict, yt1_predict = self.model.prediction_well_output_only(inputs)
        if tf.executing_eagerly():
            if return_numpy:
                return zt1_predict.numpy(), yt1_predict.numpy()
            else:
                return zt1_predict, yt1_predict
        else:
            return zt1_predict, yt1_predict

    @tf.function
    def evaluate_test(self, inputs, targets):
        xt, _, _ = inputs
        xt1, yt1 = targets

        # Forward passes:
        self.model(inputs)
        zt1 = self.use_encoder(xt1)

        # Extract:
        true = (xt, zt1, xt1, yt1)
        prediction = (self.model.xt_reconstruct, self.model.zt1_predict, self.model.xt1_predict, self.model.yt1_predict)

        self.test_loss(self.loss_e2co(true, prediction))
        self.test_rec_loss(self.loss_e2co.get_rec_loss())
        self.test_pred_loss(self.loss_e2co.get_pred_loss())
        self.test_trans_loss(self.loss_e2co.get_trans_loss())
        self.test_flux_loss(self.loss_e2co.get_flux_loss())
        self.test_well_loss(self.loss_e2co.get_well_loss())

    @tf.function
    def evaluate_val(self, inputs, targets):
        xt, _, _ = inputs
        xt1, yt1 = targets

        # Forward passes:
        self.model(inputs)
        zt1 = self.use_encoder(xt1)

        # Extract:
        true = (xt, zt1, xt1, yt1)
        prediction = (self.model.xt_reconstruct, self.model.zt1_predict, self.model.xt1_predict, self.model.yt1_predict)

        self.val_loss(self.loss_e2co(true, prediction))
        self.val_rec_loss(self.loss_e2co.get_rec_loss())
        self.val_pred_loss(self.loss_e2co.get_pred_loss())
        self.val_trans_loss(self.loss_e2co.get_trans_loss())
        self.val_flux_loss(self.loss_e2co.get_flux_loss())
        self.val_well_loss(self.loss_e2co.get_well_loss())

    @tf.function
    def update(self, inputs, targets):
        xt, _, _ = inputs
        xt1, yt1 = targets

        # Backpropagation:
        with tf.GradientTape() as tape:
            self.model(inputs)
            zt1 = self.use_encoder(xt1)
            true = (xt, zt1, xt1, yt1)
            prediction = (self.model.xt_reconstruct, self.model.zt1_predict, self.model.xt1_predict, self.model.yt1_predict)
            training_loss = self.loss_e2co(true, prediction)

        gradients = tape.gradient(training_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_total_loss(training_loss)

        self.train_rec_loss(self.loss_e2co.get_rec_loss())
        self.train_pred_loss(self.loss_e2co.get_pred_loss())
        self.train_trans_loss(self.loss_e2co.get_trans_loss())
        self.train_flux_loss(self.loss_e2co.get_flux_loss())
        self.train_well_loss(self.loss_e2co.get_well_loss())

        # self.train_rec_loss(training_loss.get_rec_loss())
        # self.train_pred_loss(training_loss.get_pred_loss())
        # self.train_trans_loss(training_loss.get_trans_loss())
        # self.train_flux_loss(training_loss.get_flux_loss())
        # self.train_well_loss(training_loss.get_well_loss())
