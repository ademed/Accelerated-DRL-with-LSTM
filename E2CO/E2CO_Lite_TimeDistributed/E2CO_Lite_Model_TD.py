import tensorflow as tf
import pickle
import numpy as np
# tf.keras.backend.set_floatx('float64')

from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import e2co_lite
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import e2co_lite_loss

from contextlib import redirect_stdout
import os

""" E2CO MODEL """
#======================================================================================================================
class e2co_lite_model():
    def __init__(self, initial_learning_rate,final_learning_rate,  input_shape, latent_dim, u_dim,
                 out_dim, weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well,
                 SWATTableFile=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\SWAT_table.txt',
                 GRIDMinMax=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\GridMinMax.txt',
                 NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm',
                 Nx=60, Ny=60, Nz=4, exclude_decoder = False, discard_decoder = False):
        if not discard_decoder:
            if exclude_decoder:               
                self.model = e2co_lite(input_shape, latent_dim, u_dim, out_dim, discard_decoder)
                #freeze decoder layers
                layer = self.model.decoder
                layer.trainable = False
                self.loss_e2co_lite = e2co_lite_loss(weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well,SWATTableFile,
                                                        GRIDMinMax,NumberOfRealizations, PermFolder, Nx, Ny, Nz, exclude_decoder, discard_decoder)
                self.train_total_loss = tf.keras.metrics.Mean(name='train/total_loss')
                self.train_trans_loss = tf.keras.metrics.Mean(name='train/trans_loss')
                self.train_well_loss = tf.keras.metrics.Mean(name='train/well_loss')

                self.val_loss = tf.keras.metrics.Mean(name='val_loss')
                self.val_trans_loss = tf.keras.metrics.Mean(name='val/trans_loss')
                self.val_well_loss = tf.keras.metrics.Mean(name='val/well_loss')

                self.test_loss = tf.keras.metrics.Mean(name='test_loss')
                self.test_trans_loss = tf.keras.metrics.Mean(name='test/trans_loss')
                self.test_well_loss = tf.keras.metrics.Mean(name='test/well_loss')
            else:
                self.model = e2co_lite(input_shape, latent_dim, u_dim, out_dim, discard_decoder)
                self.loss_e2co_lite = e2co_lite_loss(weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well,SWATTableFile,
                                                        GRIDMinMax,NumberOfRealizations, PermFolder, Nx, Ny, Nz, exclude_decoder, discard_decoder)
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
        elif discard_decoder:
            self.model = e2co_lite(input_shape, latent_dim, u_dim, out_dim, discard_decoder)
            self.loss_e2co_lite = e2co_lite_loss(weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well,exclude_decoder, discard_decoder)
            self.train_total_loss = tf.keras.metrics.Mean(name='train/total_loss')
            self.train_trans_loss = tf.keras.metrics.Mean(name='train/trans_loss')
            self.train_well_loss = tf.keras.metrics.Mean(name='train/well_loss')
            
            self.val_loss = tf.keras.metrics.Mean(name='val_loss')
            self.val_trans_loss = tf.keras.metrics.Mean(name='val/trans_loss')
            self.val_well_loss = tf.keras.metrics.Mean(name='val/well_loss')

            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_trans_loss = tf.keras.metrics.Mean(name='test/trans_loss')
            self.test_well_loss = tf.keras.metrics.Mean(name='test/well_loss')
        
        self.discard_decoder = discard_decoder
        self.exclude_decoder = exclude_decoder
        self.initial_learning_rate = initial_learning_rate  # maximum learning rate
        self.final_learning_rate = final_learning_rate   # minimum learning rate
        decay_steps = 1000           # total number of steps to reach the minimum        
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(self.initial_learning_rate, decay_steps, self.final_learning_rate, power=1.0,cycle=False) # If True, learning rate will reset to the initial value after decay_steps  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

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

            if not self.discard_decoder:
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
        zt, Et = self.model.encoder(xt)
        if tf.executing_eagerly():
            if return_numpy:
                return zt.numpy(), Et.numpy()
            else:
                return zt, Et
        else:
            return zt, Et


    def predict(self, inputs, return_numpy=True):
        zt1_predict, xt1_predict, yt1_predict = self.model.prediction(inputs)
        if tf.executing_eagerly():
            if return_numpy:
                return zt1_predict.numpy(), xt1_predict.numpy(), yt1_predict.numpy()
            else:
                return zt1_predict, xt1_predict, yt1_predict
        else:
            return zt1_predict, xt1_predict, yt1_predict

    def predict_state(self, inputs, return_numpy=True):
        zt1_predict, xt1_predict = self.model.prediction_state(inputs)
        if tf.executing_eagerly():
            if return_numpy:
                return zt1_predict.numpy(), xt1_predict.numpy()
            else:
                return zt1_predict, xt1_predict
        else:
            return zt1_predict, xt1_predict

    def predict_well_output(self, inputs, return_numpy=True):
        zt1_predict, yt1_predict = self.model.prediction_well(inputs)
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
        zt1, _ = self.use_encoder(xt1)
        
        predictions = self.model(inputs)   
        true = (xt, zt1, xt1, yt1)

        self.test_loss(self.loss_e2co_lite(true, predictions))

        self.test_rec_loss(self.loss_e2co_lite.get_rec_loss())
        self.test_pred_loss(self.loss_e2co_lite.get_pred_loss())
        self.test_trans_loss(self.loss_e2co_lite.get_trans_loss())
        self.test_flux_loss(self.loss_e2co_lite.get_flux_loss())
        self.test_well_loss(self.loss_e2co_lite.get_well_loss())

    @tf.function
    def evaluate_val(self, inputs, targets):
        
        xt, _, _ = inputs
        xt1, yt1, stop_flux_loss = targets
        zt1, _ = self.use_encoder(xt1)
        
        predictions = self.model(inputs)   
        true = (xt, zt1, xt1, yt1, stop_flux_loss)

        self.val_loss(self.loss_e2co_lite(true, predictions))
        self.val_rec_loss(self.loss_e2co_lite.get_rec_loss())
        self.val_pred_loss(self.loss_e2co_lite.get_pred_loss())
        self.val_trans_loss(self.loss_e2co_lite.get_trans_loss())
        self.val_flux_loss(self.loss_e2co_lite.get_flux_loss())
        self.val_well_loss(self.loss_e2co_lite.get_well_loss())

    @tf.function
    def update(self, inputs, targets):
        
        xt, _, _ = inputs
        xt1, yt1, stop_flux_loss = targets
    
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            zt1, _ = self.model.encoder(xt1)
            true = (xt, zt1, xt1, yt1, stop_flux_loss)
            loss = self.loss_e2co_lite(true, predictions)
    
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_total_loss(loss)
        if self.exclude_decoder or self.discard_decoder: 
            self.train_well_loss(self.loss_e2co_lite.get_well_loss())                   
            self.train_trans_loss(self.loss_e2co_lite.get_trans_loss())
        else:
            self.train_pred_loss(self.loss_e2co_lite.get_pred_loss())
            self.train_rec_loss(self.loss_e2co_lite.get_rec_loss())  
            self.train_well_loss(self.loss_e2co_lite.get_well_loss())
            self.train_flux_loss(self.loss_e2co_lite.get_flux_loss())                   
            self.train_trans_loss(self.loss_e2co_lite.get_trans_loss())
        

        
    def predict_sequential(self, wellctrl, dt):
        
        n = wellctrl.shape[0]
        state_t1_pred = np.zeros((n,self.n_ctrl,self.Nx,self.Ny,self.Nz,2))
        wellout_pred = np.zeros((n,self.n_ctrl,self.out_dim))
        
        for i_tstep in range(self.n_ctrl):
            if i_tstep == 0:
                state_t0 = np.repeat(self.state_init,n,0)
                z_t_seq, Et = self.model.encoder(state_t0)
                del state_t0
                z_t_seq = z_t_seq.numpy()
            results = self.model.prediction((z_t_seq, Et, wellctrl[:,i_tstep,:], dt[:,i_tstep,:]))
            z_t_seq = results[0].numpy()
            state_t1_pred[:, i_tstep, ...] = results[1].numpy()
            wellout_pred[:, i_tstep, ...] = results[2].numpy()   

        return state_t1_pred, wellout_pred       

    
    def predict_sequential_well(self, wellctrl, dt):
        
        n = wellctrl.shape[0]
        wellout_pred = np.zeros((n,self.n_ctrl,self.out_dim))
        
        for i_tstep in range(self.n_ctrl):
            if i_tstep == 0:
                state_t0 = np.repeat(self.state_init,n,0)
                z_t_seq, _ = self.model.encoder(state_t0)
                del state_t0
                z_t_seq = z_t_seq.numpy()
            results = self.model.prediction_well((z_t_seq, wellctrl[:,i_tstep,:], dt[:,i_tstep,:]))
            z_t_seq = results[0].numpy()
            wellout_pred[:, i_tstep, ...] = results[1].numpy()   

        return wellout_pred         
    