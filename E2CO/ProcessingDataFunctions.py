import numpy as np
import pandas as pd
import sys
import shutil
import os
import pickle
sys.path.append(os.getcwd())
from E2CO.Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()


def DataNormalizer(data):
    min_val = np.min(data)
    max_val = np.max(data)

    data_norm = (data - min_val)/(max_val - min_val)

    return data_norm, min_val, max_val

def PerWellNormalizer(data):
    data_min = data[...,:].min(axis=(0,1,2))
    data_max = data[...,:].max(axis=(0,1,2))
    data_norm = (data[...,:] - data_min)/(data_max - data_min)
    return data_norm, data_min, data_max



def PerWellDenormalizer(data_norm, data_min, data_max):
    data_reconstruct = data_norm[...,:] * (data_max - data_min) +  data_min 
    return data_reconstruct

def PerWellDenormalizer_v2(data_norm, data_min, data_max):
    data_reconstruct = np.zeros(data_norm.shape)
    num_wells = data_norm.shape[-1]
    for i in range(num_wells):
        min_val = data_min[i]
        max_val = data_max[i] 
        temp = data_norm[...,i]
        temp1 = temp * (max_val - min_val) + min_val 
        data_reconstruct[...,i] = temp1
    return data_reconstruct

def PerWellNormalizer_v2(data, data_min, data_max):
    num_wells = data.shape[-1]
    data_norm = np.zeros(data.shape)
    for i in range(num_wells):
        temp1 = (data[...,i] - data_min[i])/(data_max[i] - data_min[i])
        data_norm[..., i] = temp1
    return data_norm


def DataDenormalizer(data_norm, min_val, max_val):

    data = min_val + (max_val - min_val) * data_norm

    return data

def StateLoader(LoadFolder, filename, dtype):
    with open(os.path.join(LoadFolder, filename), 'rb') as f:
        state_train = pickle.load(f)

    state_t_train = state_train[:, :-1, ...]
    state_t1_train = state_train[:, 1:, ...]

    target_shape = (-1, state_train.shape[2], state_train.shape[3], state_train.shape[4], state_train.shape[5], state_train.shape[6])
    state_t_train = state_t_train.reshape(target_shape)
    state_t1_train = state_t1_train.reshape(target_shape)

    return state_t_train.astype(dtype), state_t1_train.astype(dtype)

def ControlLoader(LoadFolder, filename, dtype):
    with open(os.path.join(LoadFolder, filename), 'rb') as f:
        U = pickle.load(f)

    target_shape = (-1, U.shape[2], U.shape[3])

    return U.reshape(target_shape).astype(dtype)

def ControlLoader2(LoadFolder, filename, dtype):
    with open(os.path.join(LoadFolder, filename), 'rb') as f:
        U = pickle.load(f)
    return U

def WellOutputLoader(LoadFolder, filenames, dtype):
    data = []

    for filename in filenames:
        with open(os.path.join(LoadFolder, filename), 'rb') as f:
            dkm = pickle.load(f)
        target_shape = (-1, dkm.shape[2], dkm.shape[3])
        data.append(dkm.reshape(target_shape).astype(dtype))

    return np.concatenate(tuple(data), axis=-1), data

def WellOutputRenormalizer_v2(well_output_test, Nprd, TrainFolder, TestFolder, dtype):
    data = []
    WOPR = well_output_test[..., :Nprd].copy()
    WWPR = well_output_test[..., Nprd:2*Nprd].copy()
    IBHP = well_output_test[..., 2*Nprd:].copy()
    filenames = {'WOPR': WOPR, 'WWPR': WWPR, 'IBHP': IBHP}
    for output_name, value in filenames.items():
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_max.pkl" ), 'rb') as f:
            train_max = pickle.load(f)
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_min.pkl" ), 'rb') as f:
            train_min = pickle.load(f)
        with open(os.path.join(TestFolder + "//" + output_name + "_test_max.pkl" ), 'rb') as f:
            test_max = pickle.load(f)
        with open(os.path.join(TestFolder + "//" + output_name + "_test_min.pkl" ), 'rb') as f:
            test_min = pickle.load(f)
        Denorm_with_test_maxmin = value[...,:] * (test_max - test_min) +  test_min
        #Denorm_with_test_maxmin = PerWellDenormalizer_v2(value, test_min, test_max)
        renorm_to_train_maxmin = (Denorm_with_test_maxmin[...,:] - train_min)/(train_max - train_min)
        #renorm_to_train_maxmin = PerWellNormalizer_v2(Denorm_with_test_maxmin, train_min, train_max)
        target_shape = (-1, value.shape[1], value.shape[2])
        data.append(renorm_to_train_maxmin.reshape(target_shape).astype(dtype))

    return np.concatenate(tuple(data), axis=-1), data


def WellOutputDenormalizer(well_output, Nprd, TrainFolder, dtype):
    data = []
    WOPR = well_output[..., :Nprd].copy()
    WWPR = well_output[..., Nprd:2*Nprd].copy()
    IBHP = well_output[..., 2*Nprd:].copy()
    filenames = {'WOPR': WOPR, 'WWPR': WWPR, 'IBHP': IBHP}
    for output_name, value in filenames.items():
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_max.pkl" ), 'rb') as f:
            train_max = pickle.load(f)
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_min.pkl" ), 'rb') as f:
            train_min = pickle.load(f)
        Denorm_with_train_maxmin = PerWellDenormalizer(value, train_min, train_max)
        target_shape = (-1, value.shape[1], value.shape[2], value.shape[3])
        data.append(Denorm_with_train_maxmin.reshape(target_shape).astype(dtype))

    return np.concatenate(tuple(data), axis=-1), data

import tensorflow as tf
def WellOutputDenormalizer_V2(well_output, Nprd, TrainFolder, dtype):
    data = []
    WOPR = well_output[..., :Nprd]
    WWPR = well_output[..., Nprd:2*Nprd]
    IBHP = well_output[..., 2*Nprd:]
    filenames = {'WOPR': WOPR, 'WWPR': WWPR, 'IBHP': IBHP}
    for output_name, value in filenames.items():
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_max.pkl" ), 'rb') as f:
            train_max = pickle.load(f)
        with open(os.path.join(TrainFolder + "//" + output_name + "_train_min.pkl" ), 'rb') as f:
            train_min = pickle.load(f)
        Denorm_with_train_maxmin = PerWellDenormalizer(value, train_min, train_max)
        target_shape = (-1, value.shape[1], value.shape[2], value.shape[3])
        data.append(tf.cast(tf.reshape(Denorm_with_train_maxmin, target_shape), dtype))

    return tf.concat(data, axis=-1), data

def WellOutputDenormalizer_test(well_output, Nprd, TestFolder, dtype):
    data = []
    WOPR = well_output[..., :Nprd].copy()
    WWPR = well_output[..., Nprd:2*Nprd].copy()
    IBHP = well_output[..., 2*Nprd:].copy()
    filenames = {'WOPR': WOPR, 'WWPR': WWPR, 'IBHP': IBHP}
    for output_name, value in filenames.items():
        with open(os.path.join(TestFolder + "//" + output_name + "_test_max.pkl" ), 'rb') as f:
            test_max = pickle.load(f)
        with open(os.path.join(TestFolder + "//" + output_name + "_test_min.pkl" ), 'rb') as f:
            test_min = pickle.load(f)
        Denorm_with_train_maxmin = PerWellDenormalizer(value, test_min, test_max)
        target_shape = (-1, value.shape[1], value.shape[2], value.shape[3])
        data.append(Denorm_with_train_maxmin.reshape(target_shape).astype(dtype))

    return np.concatenate(tuple(data), axis=-1), data

def WellOutputRenormalize(well_output_test, Nprd, TrainMinMax, TestMinMax):
    WOPR = well_output_test[:, :, :Nprd].copy()
    WWPR = well_output_test[:, :, Nprd:2*Nprd].copy()
    IBHP = well_output_test[:, :, 2*Nprd:].copy()

    with open(TrainMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    with open(TestMinMax, 'r') as file:
        test_limits = []
        dkm = file.readline()
        while dkm != '':
            test_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    WOPR_renorm = (WOPR * (test_limits[1] - test_limits[0]) + test_limits[0] - train_limits[0]) / (train_limits[1] - train_limits[0])
    WWPR_renorm = (WWPR * (test_limits[3] - test_limits[2]) + test_limits[2] - train_limits[2]) / (train_limits[3] - train_limits[2])
    #IBHP_renorm = (IBHP * (test_limits[7] - test_limits[6]) + test_limits[6] - train_limits[6]) / (train_limits[7] - train_limits[6])
    IBHP_renorm = (IBHP * (test_limits[5] - test_limits[4]) + test_limits[4] - train_limits[4]) / (train_limits[5] - train_limits[4])

    data = (WOPR_renorm, WWPR_renorm, IBHP_renorm)

    well_output_renorm = np.concatenate(data, axis=-1)

    return well_output_renorm, data

def GridRenormalize(state_test, TrainMinMax, TestMinMax):
    PRES = state_test[..., 0]
    SWAT = state_test[..., 1]

    with open(TrainMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    with open(TestMinMax, 'r') as file:
        test_limits = []
        dkm = file.readline()
        while dkm != '':
            test_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    PRES_renorm = (PRES * (test_limits[1] - test_limits[0]) + test_limits[0] - train_limits[0]) / (train_limits[1] - train_limits[0])
    #SWAT_renorm = (SWAT * (test_limits[3] - test_limits[2]) + test_limits[2] - train_limits[2]) / (train_limits[3] - train_limits[2])

    return np.concatenate((PRES_renorm[..., np.newaxis], SWAT[..., np.newaxis]), axis=-1)

def GridRenormalize_test(state_test, TrainMinMax, TestMinMax):
    PRES = state_test[..., 0]
    SWAT = state_test[..., 1]

    with open(TrainMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    with open(TestMinMax, 'r') as file:
        test_limits = []
        dkm = file.readline()
        while dkm != '':
            test_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    PRES_renorm = PRES * (test_limits[1] - test_limits[0]) + test_limits[0] 
    SWAT_renorm = SWAT * (test_limits[3] - test_limits[2]) + test_limits[2] 

    return np.concatenate((PRES_renorm[..., np.newaxis], SWAT[..., np.newaxis]), axis=-1)

def GetMinMax(TrainMinMax, TestMinMax):
    with open(TrainMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    with open(TestMinMax, 'r') as file:
        test_limits = []
        dkm = file.readline()
        while dkm != '':
            test_limits.append(float(dkm.split()[2]))
            dkm = file.readline()
    
    return train_limits, test_limits
