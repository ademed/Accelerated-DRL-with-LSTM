import random
import timeit
import os
import h5py
import numpy as np
import time

from E2CO.E2CO_Layers import *
from E2CO.E2CO_Loss2 import *
from E2CO.E2CO_Blocks2_ReLU import *
from E2CO.E2CO_Model2_ReLU import *

import pickle
from E2CO.ProcessingDataFunctions import StateLoader, ControlLoader, WellOutputLoader, WellOutputRenormalize, GridRenormalize

FLOAT_TYPE = 'float32'
#
# ############################################################################################################################################
# def SequentialPrediction(model, U_test, state_t_test_renorm, n_welloutputs=11):
#     U_test_reshaped = U_test.reshape((-1, 20, 8))
#     state_t_test_reshaped = state_t_test_renorm.reshape((-1, 20, 5, 60, 60, 4, 2))
#
#     n_data = U_test_reshaped.shape[0]
#     state_t1_seq = np.zeros(state_t_test_reshaped.shape, dtype=FLOAT_TYPE)
#     yt1_seq = np.zeros((state_t_test_reshaped.shape[0], state_t_test_reshaped.shape[1], state_t_test_reshaped.shape[2], n_welloutputs), dtype=FLOAT_TYPE)
#
#     for i in range(n_data):
#         print('Processing Sequential Prediction for data #' + str(i + 1) + '...')
#         for t in range(U_test_reshaped.shape[1]):
#             if t == 0:
#                 zt = model.use_encoder(np.expand_dims(state_t_test_reshaped[i, t, ...], axis=0))
#
#             dt = np.ones((1, 1), dtype=FLOAT_TYPE)
#             ut1 = np.expand_dims(U_test_reshaped[i, t, :], axis=0)
#             zt1, xt1, yt1 = model.predict((zt, ut1, dt))
#
#             state_t1_seq[i, t, ...] = xt1.copy()
#             yt1_seq[i, t, ...] = yt1.copy()
#
#             zt = zt1.copy()
#
#     return state_t1_seq, yt1_seq

############################################################################################################################################
def ForwardPrediction(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE)

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if U_test.ndim < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if state_init.ndim < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    state_t1_seq = []
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, xt1, yt1 = model.predict((zt_list[t], ut1, dt), return_numpy=False)

        state_t1_seq.append(tf.expand_dims(xt1, axis=1))
        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    state_t1_seq_out = tf.concat(state_t1_seq, axis=1)
    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

    if return_numpy:
        return state_t1_seq_out.numpy(), yt1_seq_out.numpy()
    else:
        return state_t1_seq_out, yt1_seq_out
    

  
def ForwardPrediction_WellOutputOnly(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.Variable(tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE))

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if (tf.rank(U_test)).numpy() < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if (tf.rank(state_init)).numpy() < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, yt1 = model.predict_output((zt_list[t], ut1, dt), return_numpy=False)

        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))
    whocares = []
    if return_numpy:
        return whocares, yt1_seq_out.numpy()
    else:
        return whocares, yt1_seq_out

#@tf.function
def ForwardPrediction_WellOutputOnly_tf(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):
    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.Variable(tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE))

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if len(U_test.shape) < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    #n_data = U_test_reshaped.shape[0]
    #n_data = tf.convert_to_tensor(n_data, dtype=tf.variant)
    n_data = 1

    if len(state_init.shape) < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, yt1 = model.predict_output((zt_list[t], ut1, dt), return_numpy=False)

        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))
    whocares = []
    if return_numpy:
        return whocares, yt1_seq_out.numpy()
    else:
        return whocares, yt1_seq_out

def ForwardPrediction_LITE(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE)

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if U_test.ndim < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, NE, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if state_init.ndim < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    state_t1_seq = []
    yt1_seq = []
    zt_list = []
    Et_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt, Et = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)
            

        dt = tf.ones((n_data, NE, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, ...]

        zt1, xt1, yt1 = model.predict((zt_list[t], Et, ut1, dt), return_numpy=False)

        state_t1_seq.append(tf.expand_dims(xt1, axis=1))
        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    state_t1_seq_out = tf.concat(state_t1_seq, axis=1)
    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

    if return_numpy:
        return state_t1_seq_out.numpy(), yt1_seq_out.numpy()
    else:
        return state_t1_seq_out, yt1_seq_out
    
@tf.function
def ForwardPrediction_LITE_v3(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True, batch_size=100):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :param batch_size: Number of samples to process per batch (default 100)
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE)

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    
    # Reshape preprocessing:
    U_test_reshaped = tf.reshape(U_test, (-1, n_controls, 1, n_controls_per_well))
    U_test_reshaped = tf.tile(U_test_reshaped, [1, 1, NE, 1])
    n_data = 450 #U_test_reshaped.shape[0]

    if len(state_init.shape) < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data.numpy(), 1, 1, 1, 1, 1]))

    # Initialization:
    state_t1_seq = []
    yt1_seq = []

    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints in batches of ' + str(batch_size) + '...')
    
    # Batch processing
    for start in range(0, n_data, batch_size):
        end = min(start + batch_size, n_data)
        batch_U_test = U_test_reshaped[start:end, ...]
        batch_state_init = state_init_reshaped[start:end, ...]

        # Passing to the E2CO model:
        batch_state_t1_seq = []
        batch_yt1_seq = []
        zt_list = []
        
        for t in range(n_controls):
            if t == 0:
                zt, Et = model.use_encoder(batch_state_init, return_numpy=False)
                zt_list.append(zt)

            dt = tf.ones((end - start, NE, 1), dtype=FLOAT_TYPE)
            ut1 = batch_U_test[:, t, ...]

            zt1, xt1, yt1 = model.predict((zt_list[t], Et, ut1, dt), return_numpy=False)

            batch_state_t1_seq.append(tf.expand_dims(xt1, axis=1))
            batch_yt1_seq.append(tf.expand_dims(yt1, axis=1))
            zt_list.append(zt1)

        # Concatenate outputs for the batch
        state_t1_seq.append(tf.concat(batch_state_t1_seq, axis=1))
        yt1_seq.append(tf.concat(batch_yt1_seq, axis=1))

    # Combine all batches
    state_t1_seq_out = tf.concat(state_t1_seq, axis=0)
    yt1_seq_out = tf.concat(yt1_seq, axis=0)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

    if return_numpy:
        return state_t1_seq_out.numpy(), yt1_seq_out.numpy()
    else:
        return state_t1_seq_out, yt1_seq_out


def ForwardPrediction_LITE_v2(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True, batch_size=100):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :param batch_size: Number of samples to process per batch (default 100)
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()


    # Reshape preprocessing:
    if U_test.ndim < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, NE, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if state_init.ndim < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))


    # Initialization:
    state_t1_seq = []
    yt1_seq = []

    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints in batches of ' + str(batch_size) + '...')
    
    # Batch processing
    for start in range(0, n_data, batch_size):
        end = min(start + batch_size, n_data)
        batch_U_test = U_test_reshaped[start:end, ...]
        batch_state_init = state_init_reshaped[start:end, ...]

        # Passing to the E2CO model:
        batch_state_t1_seq = []
        batch_yt1_seq = []
        zt_list = []
        
        for t in range(n_controls):
            if t == 0:
                zt, Et = model.use_encoder(batch_state_init, return_numpy=False)
                zt_list.append(zt)

            dt = tf.ones((end - start, NE, 1), dtype=FLOAT_TYPE)
            ut1 = batch_U_test[:, t, ...]

            zt1, xt1, yt1 = model.predict((zt_list[t], Et, ut1, dt), return_numpy=False)

            batch_state_t1_seq.append(tf.expand_dims(xt1, axis=1))
            batch_yt1_seq.append(tf.expand_dims(yt1, axis=1))
            zt_list.append(zt1)

        # Concatenate outputs for the batch
        state_t1_seq.append(tf.concat(batch_state_t1_seq, axis=1))
        yt1_seq.append(tf.concat(batch_yt1_seq, axis=1))

    # Combine all batches
    state_t1_seq_out = tf.concat(state_t1_seq, axis=0)
    yt1_seq_out = tf.concat(yt1_seq, axis=0)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

    if return_numpy:
        return state_t1_seq_out.numpy(), yt1_seq_out.numpy()
    else:
        return state_t1_seq_out, yt1_seq_out

@tf.function
def ForwardPrediction_WellOutputOnly_LITE_tf(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):
    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.Variable(tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE))

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    U_test_reshaped = tf.reshape(U_test, (-1, n_controls, 1, n_controls_per_well))
    U_test_reshaped = tf.tile(U_test_reshaped, [1, 1, NE, 1])
    #n_data = U_test_reshaped.shape[0]
    #n_data = tf.convert_to_tensor(n_data, dtype=tf.variant)
    n_data = 1

    if len(state_init.shape) < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt,_ = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data,NE, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, ...]

        zt1, yt1 = model.predict_well_output((zt_list[t], ut1, dt), return_numpy=False)

        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))
    whocares = []
    if return_numpy:
        return whocares, yt1_seq_out.numpy()
    else:
        return whocares, yt1_seq_out
     
@tf.function
def ForwardPrediction_WellOutputOnly_tf2(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):
    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.Variable(tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE))

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if len(U_test.shape) < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]
    


    if len(state_init.shape) < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, yt1 = model.predict_output((zt_list[t], ut1, dt), return_numpy=False)

        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))
    whocares = []
    if return_numpy:
        return whocares, yt1_seq_out.numpy()
    else:
        return whocares, yt1_seq_out
    
def ForwardPrediction_tf(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):
    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE)

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if U_test.ndim < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if state_init.ndim < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), [n_data, 1, 1, 1, 1, 1])
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))


    # Initialization:
    #state_t1_seq = tf.TensorArray(tf.float32, size=n_controls)
    yt1_seq = tf.TensorArray(FLOAT_TYPE, size=n_controls)
    zt_list = tf.TensorArray(FLOAT_TYPE, size=n_controls)

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in tf.range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list = zt_list.write(t, zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, xt1, yt1 = model.predict((zt_list.read(t), ut1, dt), return_numpy=False)

        #state_t1_seq = state_t1_seq.write(t, tf.expand_dims(xt1, axis=1))
        yt1_seq = yt1_seq.write(t, tf.expand_dims(yt1, axis=1))
        if t >0:
            zt_list = zt_list.write(t, zt1)

    #state_t1_seq_out = state_t1_seq.concat()
    yt1_seq_out = yt1_seq.concat()
    desired_shape = tf.stack([-1, n_controls, NE, n_welloutputs])
    reshaped_yt1_seq_out = tf.reshape(yt1_seq_out, desired_shape)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1 - t0)))

    if return_numpy:
        return  reshaped_yt1_seq_out.numpy()
    else:
        return reshaped_yt1_seq_out
    
def ForwardPrediction_tf2(model, U_test, state_init, n_welloutputs=11, n_controls_per_well=8, n_controls=20, NE=5, Nx=60, Ny=60, Nz=4, N_states=2, return_numpy=True, time_simulation=True):

    '''
    :param model: DL model to be passed to
    :param U_test: tf.Tensor of shape (N_data, N_control_steps, N_wells) or (N_data * N_control_steps, N_wells)
    :param state_init: tf.Tensor of shape (NE, NX, NY, NZ, 2) or (1, NE, NX, NY, NZ, 2)
    :param n_welloutputs: number of well outputs
    :param n_controls_per_well: number of controls per wells
    :param n_controls: number of control time steps
    :param NE: number of geological realizations
    :param Nx: number of grids in x-direction
    :param Ny: number of grids in y-direction
    :param Nz: number of grids in z-direction
    :param N_states: number of state variables (2 for oil-water)
    :param return_numpy: Boolean variable whether the user wants to return the final output as a numpy ndarray. If False (default), tf.tensor will be returned
    :param time_simulation: Boolean variable indicating whether the user wants to print time elapsed
    :return: [state_t1_seq, yt1_seq] as the predicted states and the predicted outputs, respectively
    '''

    if time_simulation:
        import time
        t0 = time.time()

    # Convert from np.ndarray to tf.tensor if the user input np.ndarrays:
    if isinstance(U_test, np.ndarray):
        U_test = tf.convert_to_tensor(U_test, dtype=FLOAT_TYPE)

    if isinstance(state_init, np.ndarray):
        state_init = tf.convert_to_tensor(state_init, dtype=FLOAT_TYPE)

    # Reshape preprocessing:
    if U_test.ndim < 3:
        U_test_reshaped = tf.reshape(U_test, (-1, n_controls, n_controls_per_well))
    else:
        U_test_reshaped = U_test

    n_data = U_test_reshaped.shape[0]

    if state_init.ndim < 6:
        state_init_reshaped = tf.tile(tf.reshape(state_init, (1, NE, Nx, Ny, Nz, N_states)), tf.constant([n_data, 1, 1, 1, 1, 1]))
    else:
        state_init_reshaped = tf.tile(state_init, tf.constant([n_data, 1, 1, 1, 1, 1]))

    # Initialization:
    state_t1_seq = []
    yt1_seq = []
    zt_list = []

    # Passing to the E2CO model:
    print('Processing Sequential Prediction for ' + str(n_data) + ' datapoints...')
    for t in range(n_controls):
        if t == 0:
            zt = model.use_encoder(state_init_reshaped, return_numpy=False)
            zt_list.append(zt)

        dt = tf.ones((n_data, 1), dtype=FLOAT_TYPE)
        ut1 = U_test_reshaped[:, t, :]

        zt1, xt1, yt1 = model.predict((zt_list[t], ut1, dt), return_numpy=False)

        state_t1_seq.append(tf.expand_dims(xt1, axis=1))
        yt1_seq.append(tf.expand_dims(yt1, axis=1))
        zt_list.append(zt1)

    state_t1_seq_out = tf.concat(state_t1_seq, axis=1)
    yt1_seq_out = tf.concat(yt1_seq, axis=1)

    if time_simulation:
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

    if return_numpy:
        return yt1_seq_out.numpy()
    else:
        return yt1_seq_out

def OutputDenormalize(yt1_seq, WellRatesMinMax=r'S:\ENS\Petroleum Engineering\TUPREP\common\QuangNguyen\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\WellRatesMinMax.txt'):

    if isinstance(yt1_seq, tf.Tensor):
        yt1_seq = yt1_seq.numpy()

    WOPR = yt1_seq[..., :3].copy()
    WWPR = yt1_seq[..., 3:6].copy()
    IBHP = yt1_seq[..., 6:].copy()

    with open(WellRatesMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    WOPR_denorm = WOPR * (train_limits[1] - train_limits[0]) + train_limits[0]
    WWPR_denorm = WWPR * (train_limits[3] - train_limits[2]) + train_limits[2]
    IBHP_denorm = IBHP * (train_limits[7] - train_limits[6]) + train_limits[6]

    #return WOPR_denorm.astype(np.float64), WWPR_denorm.astype(np.float64), IBHP_denorm.astype(np.float64)
    return WOPR_denorm, WWPR_denorm, IBHP_denorm

#@tf.function
## Tensorflow implementation of OutputDenormalize for automatic differentiation
def OutputDenormalize_tf(yt1_seq, WellRatesMinMax):
    WOPR = yt1_seq[..., :3]
    WWPR = yt1_seq[..., 3:6]
    IBHP = yt1_seq[..., 6:]

    with open(WellRatesMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    WOPR_denorm = WOPR * (train_limits[1] - train_limits[0]) + train_limits[0]
    WWPR_denorm = WWPR * (train_limits[3] - train_limits[2]) + train_limits[2]
    IBHP_denorm = IBHP * (train_limits[7] - train_limits[6]) + train_limits[6]

    return WOPR_denorm, WWPR_denorm, IBHP_denorm

@tf.function
def OutputDenormalize_tf2(yt1_seq, WellRatesMinMax):
    #print(yt1_seq.shape)
    WOPR = yt1_seq[..., :3]
    WWPR = yt1_seq[..., 3:6]
    IBHP = yt1_seq[..., 6:]

    with open(WellRatesMinMax, 'r') as file:
        train_limits = []
        dkm = file.readline()
        while dkm != '':
            train_limits.append(float(dkm.split()[2]))
            dkm = file.readline()

    WOPR_denorm = WOPR * (train_limits[1] - train_limits[0]) + train_limits[0]
    WWPR_denorm = WWPR * (train_limits[3] - train_limits[2]) + train_limits[2]
    IBHP_denorm = IBHP * (train_limits[7] - train_limits[6]) + train_limits[6]

    return WOPR_denorm, WWPR_denorm, IBHP_denorm

def ControlMat2Vec(U_mat, Ninj):
    if isinstance(U_mat, tf.Tensor):
        U_mat = U_mat.numpy()

    if U_mat.ndim == 3:
        u_vec = U_mat.reshape((U_mat.shape[0], -1), order='F')
        rates = U_mat[..., :Ninj].reshape((U_mat.shape[0], -1), order='F')
        BHPs = U_mat[..., Ninj:].reshape((U_mat.shape[0], -1), order='F')
    else:
        u_vec = U_mat.reshape((-1, 1), order='F')
        rates = U_mat[..., :Ninj].reshape((-1, 1), order='F')
        BHPs = U_mat[..., Ninj:].reshape((-1, 1), order='F')

    return u_vec, rates, BHPs


def ControlVec2Mat(u_vec, n_cyc):
    if isinstance(u_vec, tf.Tensor):
        u_vec1 = u_vec.numpy().copy()
    else:
        u_vec1 = u_vec.copy()

    if u_vec.shape[-1] > 1:
        U_mat = u_vec1.reshape((u_vec.shape[0], n_cyc, -1) , order='F')
    else:
        U_mat = u_vec1.reshape((n_cyc, -1), order='F')

    return U_mat


def GetFWIR(U_mat, Ninj, Bounds, ControlStep=60, keepdims=False):
    if isinstance(U_mat, tf.Tensor):
        U_mat1 = U_mat.numpy().copy()
    else:
        U_mat1 = U_mat.copy()

    rates = U_mat1[..., :Ninj]
    WWIR_denorm = rates * (Bounds.UpperWinj - Bounds.LowerWinj) + Bounds.LowerWinj
    FWIR = np.sum(WWIR_denorm, axis=-1, keepdims=keepdims)
   ## FWIR = np.repeat(FWIR[:,:, np.newaxis], 5, axis = 2)  ###hardcorded number of realization = 5

    return WWIR_denorm.astype(np.float64), FWIR.astype(np.float64)

@tf.function
def GetFWIR_tf2(U_mat, Ninj, Bounds, n_controls, n_controls_per_well, keepdims=False):
    if len(U_mat.shape) < 3:
        U_mat = tf.reshape(U_mat, (-1, n_controls, n_controls_per_well)) ##this might lead to trouble
    else:
        U_mat = U_mat
    rates = U_mat[..., :Ninj]
    WWIR_denorm = rates * (Bounds.UpperWinj - Bounds.LowerWinj) + Bounds.LowerWinj
    FWIR = tf.reduce_sum(WWIR_denorm, axis=-1, keepdims=keepdims)

    return WWIR_denorm, FWIR

def GetFWIR_tf(U_mat, Ninj, Bounds, n_controls, n_controls_per_well, keepdims=False):
    if len(U_mat.shape) < 3:
        U_mat = tf.reshape(U_mat, (-1, n_controls, n_controls_per_well))
    else:
        U_mat = U_mat
    rates = U_mat[..., :Ninj]
    WWIR_denorm = rates * (Bounds.UpperWinj - Bounds.LowerWinj) + Bounds.LowerWinj
    FWIR = tf.reduce_sum(WWIR_denorm, axis=-1, keepdims=keepdims)

    return WWIR_denorm, FWIR

def GetFieldInjection(FWIR, IBHP):
    InjData = {'FWIR': FWIR,
               'IBHP':IBHP}

    return InjData

def GetFieldProductionRates(WOPR_denorm, WWPR_denorm, ControlStep=60, keepdims=False):
    FOPR = np.sum(WOPR_denorm, axis=-1, keepdims=keepdims)
    FWPR = np.sum(WWPR_denorm, axis=-1, keepdims=keepdims)
    FLPR = FOPR + FWPR

    PrdData = {'FLPR': FLPR,
               'FOPR': FOPR,
                'FWPR': FWPR}

    return FOPR, FWPR, FLPR, PrdData

@tf.function
## Tensorflow implementation of GetFieldProductionRates function for automatic differentiation
def GetFieldProductionRates_tf2(WOPR_denorm, WWPR_denorm, ControlStep=60, keepdims=False):
    FOPR = tf.reduce_sum(WOPR_denorm, axis=-1, keepdims=keepdims)
    FWPR = tf.reduce_sum(WWPR_denorm, axis=-1, keepdims=keepdims)
    FLPR = FOPR + FWPR

    PrdData = {'FLPR': FLPR,
               'FOPR': FOPR,
               'FWPR': FWPR}

    return FOPR, FWPR, FLPR, PrdData

def GetFieldProductionRates_tf(WOPR_denorm, WWPR_denorm, ControlStep=60, keepdims=False):
    FOPR = tf.reduce_sum(WOPR_denorm, axis=-1, keepdims=keepdims)
    FWPR = tf.reduce_sum(WWPR_denorm, axis=-1, keepdims=keepdims)
    FLPR = FOPR + FWPR

    PrdData = {'FLPR': FLPR,
               'FOPR': FOPR,
               'FWPR': FWPR}

    return FOPR, FWPR, FLPR, PrdData
