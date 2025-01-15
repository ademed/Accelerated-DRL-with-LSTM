import numpy as np
import os
from scipy import interpolate

from keras import backend as K
from keras.losses import Loss

import matplotlib.pyplot as plt
import tensorflow as tf

FLOAT_TYPE = 'float32'


class RelPerm():
    def __init__(self, SWATTableFile=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\SWAT_table.txt',
                 GRIDMinMax=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\GridMinMax.txt',
                 NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm',
                 Nx=60, Ny=60, Nz=4):
        gridminmaxfile = open(GRIDMinMax, 'r')
        dkm = gridminmaxfile.readlines()
        gridminmaxfile.close()

        self.SWAT_min = np.float32(dkm[2].replace('SWAT_min =', '').strip()) #SWAT_min is replaced with an empty string and strip() removes trailing and leading white spaces
        self.SWAT_max = np.float32(dkm[3].replace('SWAT_max =', '').strip())

        self.kr_data = np.loadtxt(SWATTableFile).astype(FLOAT_TYPE)
        self.krw_func = interpolate.interp1d(self.kr_data[:, 0], self.kr_data[:, 1], kind='cubic', bounds_error=False, fill_value=(self.kr_data[0, 1], self.kr_data[-1, 1]))
        self.kro_func = interpolate.interp1d(self.kr_data[:, 0], self.kr_data[:, 2], kind='cubic', bounds_error=False, fill_value=(self.kr_data[0, 2], self.kr_data[-1, 2]))

        self.get_absolute_perm(NumberOfRealizations, PermFolder, Nx, Ny, Nz)

    def get_absolute_perm(self, NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm', Nx=60, Ny=60, Nz=4):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.permx = np.zeros((NumberOfRealizations, self.Nx, self.Ny, self.Nz), dtype=FLOAT_TYPE)
        self.permy = np.zeros((NumberOfRealizations, self.Nx, self.Ny, self.Nz), dtype=FLOAT_TYPE)
        self.permz = np.zeros((NumberOfRealizations, self.Nx, self.Ny, self.Nz), dtype=FLOAT_TYPE)

        for Realization in range(NumberOfRealizations):
            filepathx = os.path.join(PermFolder + "\\permx_real" + str(Realization + 1) + ".inc")
            filepathy = os.path.join(PermFolder + "\\permy_real" + str(Realization + 1) + ".inc")
            filepathz = os.path.join(PermFolder + "\\permz_real" + str(Realization + 1) + ".inc")

            self.permx[Realization, : , :, :] = np.loadtxt(filepathx, skiprows=1).astype(FLOAT_TYPE).reshape((self.Nz, self.Ny, self.Nx)).transpose()
            self.permy[Realization, :, :, :] = np.loadtxt(filepathy, skiprows=1).astype(FLOAT_TYPE).reshape((self.Nz, self.Ny, self.Nx)).transpose()
            self.permz[Realization, :, :, :] = np.loadtxt(filepathz, skiprows=1).astype(FLOAT_TYPE).reshape((self.Nz, self.Ny, self.Nx)).transpose()

        # Convert to TF32 Tensors with an extra dimension in front of the original arrays for the batch size:
        self.permx = tf.expand_dims(self.permx, axis=0)
        self.permy = tf.expand_dims(self.permy, axis=0)
        self.permz = tf.expand_dims(self.permz, axis=0)

    def denormalizer(self, SWAT):
        return self.SWAT_min + (self.SWAT_max - self.SWAT_min) * SWAT

    def get_krw(self, SWAT):
        return self.krw_func(self.denormalizer(SWAT)).astype(FLOAT_TYPE)

    def get_kro(self, SWAT):
        return self.kro_func(self.denormalizer(SWAT)).astype(FLOAT_TYPE)

    def get_k(self):
        return self.permx, self.permy, self.permz



def loss_sql2norm(x_true, x_pred):
    return K.mean(K.sum((K.batch_flatten(x_true) - K.batch_flatten(x_pred)) ** 2, axis=-1))

def loss_l2norm(x_true, x_pred):
    return K.mean(K.sqrt(K.sum((K.batch_flatten(x_true) - K.batch_flatten(x_pred)) ** 2, axis=-1)))

def loss_l1norm(x_true, x_pred):
    return K.mean(K.abs(K.batch_flatten(x_true) - K.batch_flatten(x_pred)) , axis=-1)
def loss_flux(state_true, state_pred,
              SWATTableFile=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\SWAT_table.txt',
                 GRIDMinMax=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\GridMinMax.txt',
                 NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm',
                 Nx=60, Ny=60, Nz=4):

    # State variables: dimensions = N x NE x Nx x Ny x Nz x 2
    PRES_true = state_true[:, :, :, :, :, 0]
    PRES_pred = state_pred[:, :, :, :, :, 0]

    SWAT_true = state_true[:, :, :, :, :, 1]
    SWAT_pred = state_pred[:, :, :, :, :, 1]

    k_kr = RelPerm(SWATTableFile, GRIDMinMax, NumberOfRealizations, PermFolder, Nx, Ny, Nz)
    kx, ky, kz = k_kr.get_k()

    # Harmonic averages:
    tran_x = 1./kx[:, :, 1:, ...] + 1./kx[:, :, :-1, ...]
    tran_y = 1./ky[:, :, :, 1:, ...] + 1./ky[:, :, :, :-1, ...]
    tran_z = 1./kz[:, :, :, :, 1:] + 1./kz[:, :, :, :, :-1]

    # Calculate oil-phase fluxes:
    flux_x_o_true = (PRES_true[:, :, 1:, :, :] - PRES_true[:, :, :-1, :, :]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_true[:, :, 1:, :, :]], Tout=tf.float32)
    flux_x_o_pred = (PRES_pred[:, :, 1:, :, :] - PRES_pred[:, :, :-1, :, :]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_pred[:, :, 1:, :, :]], Tout=tf.float32)

    flux_y_o_true = (PRES_true[:, :, :, 1:, :] - PRES_true[:, :, :, :-1, :]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_true[:, :, :, 1:, :]], Tout=tf.float32)
    flux_y_o_pred = (PRES_pred[:, :, :, 1:, :] - PRES_pred[:, :, :, :-1, :]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_pred[:, :, :, 1:, :]], Tout=tf.float32)

    flux_z_o_true = (PRES_true[:, :, :, :, 1:] - PRES_true[:, :, :, :, :-1]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_true[:, :, :, :, 1:]], Tout=tf.float32)
    flux_z_o_pred = (PRES_pred[:, :, :, :, 1:] - PRES_pred[:, :, :, :, :-1]) * tf.numpy_function(func=k_kr.get_kro, inp=[SWAT_pred[:, :, :, :, 1:]], Tout=tf.float32)

    # Calculate water-phase fluxes:
    flux_x_w_true = (PRES_true[:, :, 1:, :, :] - PRES_true[:, :, :-1, :, :]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_true[:, :, 1:, :, :]], Tout=tf.float32)
    flux_x_w_pred = (PRES_pred[:, :, 1:, :, :] - PRES_pred[:, :, :-1, :, :]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_pred[:, :, 1:, :, :]], Tout=tf.float32)

    flux_y_w_true = (PRES_true[:, :, :, 1:, :] - PRES_true[:, :, :, :-1, :]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_true[:, :, :, 1:, :]], Tout=tf.float32)
    flux_y_w_pred = (PRES_pred[:, :, :, 1:, :] - PRES_pred[:, :, :, :-1, :]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_pred[:, :, :, 1:, :]], Tout=tf.float32)

    flux_z_w_true = (PRES_true[:, :, :, :, 1:] - PRES_true[:, :, :, :, :-1]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_true[:, :, :, :, 1:]], Tout=tf.float32)
    flux_z_w_pred = (PRES_pred[:, :, :, :, 1:] - PRES_pred[:, :, :, :, :-1]) * tf.numpy_function(func=k_kr.get_krw, inp=[SWAT_pred[:, :, :, :, 1:]], Tout=tf.float32)

    # Calculate total flux loss for the whole minibatch:
    flux_x = K.batch_flatten(K.abs(flux_x_o_true-flux_x_o_pred)/tran_x) + K.batch_flatten(K.abs(flux_x_w_true-flux_x_w_pred)/tran_x)
    flux_y = K.batch_flatten(K.abs(flux_y_o_true-flux_y_o_pred)/tran_y) + K.batch_flatten(K.abs(flux_y_w_true-flux_y_w_pred)/tran_y)
    flux_z = K.batch_flatten(K.abs(flux_z_o_true-flux_z_o_pred)/tran_z) + K.batch_flatten(K.abs(flux_z_w_true-flux_z_w_pred)/tran_z)

    # Average L2 norm of the flux loss:
    return K.mean(K.sqrt(K.sum(flux_x**2, axis=-1) + K.sum(flux_y**2, axis=-1) + K.sum(flux_z**2, axis=-1)))

""" E2CO_Lite LOSS """
#======================================================================================================================
class e2co_lite_loss(Loss):
    def __init__(self, weight_loss_rec_pred, weight_loss_trans, weight_loss_flux, weight_loss_well, 
                 SWATTableFile=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\SWAT_table.txt',
                 GRIDMinMax=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\GridMinMax.txt',
                 NumberOfRealizations=5, PermFolder=r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CODataset_BHP3300to3800_Rate800to1500\AbsolutePerm',
                 Nx=60, Ny=60, Nz=4, exclude_decoder = False, discard_decoder = False):
        
        super(e2co_lite_loss, self).__init__()

        self.weight_loss_rec_pred = weight_loss_rec_pred
        self.weight_loss_trans = weight_loss_trans 
        self.weight_loss_flux = weight_loss_flux 
        self.weight_loss_well = weight_loss_well 

        self.exclude_decoder = exclude_decoder
        self.discard_decoder = discard_decoder
        

        self.rec_loss = None
        self.pred_loss = None
        self.trans_loss = None
        self.total_flux_loss = None
        self.well_loss =  None
        self.total_loss = None 
        self.SWATTableFile = SWATTableFile
        self.GRIDMinMax = GRIDMinMax
        self.PermFolder = PermFolder
        self.NE = NumberOfRealizations
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
    
    def call(self, true, pred):
        xt, zt1, xt1, yt1, stop_flux_loss = true   ## I added the stop_flux_loss boolean here because I needed a quick fix..adding withing the model was not running as expected.
        if not self.discard_decoder:
            xt_rec, zt1_pred, xt1_pred, yt1_pred = pred  
        elif self.discard_decoder:
            zt1_pred, yt1_pred = pred           
        
        if self.exclude_decoder or self.discard_decoder:                   
            ##-------WELL OUTPUT LOSS
            self.well_loss = loss_l2norm(yt1, yt1_pred) * self.weight_loss_well
            ##-------TRANSITION LOSS
            self.trans_loss = loss_l2norm(zt1, zt1_pred) * self.weight_loss_trans   
            ##-------TOTAL LOSS
            self.total_loss = self.well_loss + self.trans_loss        
        else:
            ##-------WELL OUTPUT LOSS
            self.well_loss = loss_l2norm(yt1, yt1_pred) * self.weight_loss_well
            ##-------TRANSITION LOSS
            self.trans_loss = loss_l2norm(zt1, zt1_pred) * self.weight_loss_trans         
            ##-------STATE PREDICTION LOSS
            self.pred_loss = loss_l2norm(xt1, xt1_pred) * self.weight_loss_rec_pred
            ##-------RECONSTRUCTION LOSS
            self.rec_loss = loss_l2norm(xt, xt_rec) * self.weight_loss_rec_pred 
            ##-------FLUX LOSS
            if stop_flux_loss == True:
                self.total_flux_loss = 0.001
            else:
                flux_rec_loss = loss_flux(xt, xt_rec, SWATTableFile=self.SWATTableFile, GRIDMinMax=self.GRIDMinMax, NumberOfRealizations=self.NE,
                                    PermFolder=self.PermFolder, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)
                flux_pred_loss = loss_flux(xt1, xt1_pred, SWATTableFile=self.SWATTableFile, GRIDMinMax=self.GRIDMinMax, NumberOfRealizations=self.NE,
                                    PermFolder=self.PermFolder, Nx=self.Nx, Ny=self.Ny, Nz=self.Nz)
                self.total_flux_loss = (flux_rec_loss + flux_pred_loss) * self.weight_loss_flux 
                
            ##--------TOTAL LOSS
            self.total_loss = self.pred_loss + self.well_loss +  self.rec_loss + self.trans_loss + self.total_flux_loss
        
        return self.total_loss
 
    def get_rec_loss(self):
        return self.rec_loss   

    def get_pred_loss(self):
        return self.pred_loss

    def get_trans_loss(self):
        return self.trans_loss

    def get_flux_loss(self):
        return self.total_flux_loss
    
    def get_well_loss(self):
        return self.well_loss
    
    def get_total_loss(self):
        return self.total_loss
    
    