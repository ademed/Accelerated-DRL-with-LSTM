import copy
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, PReLU, LSTM
import numpy as np
import sys, os
import random

sys.path.append(os.getcwd())
from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Model_TD import *
from E2CO.ProcessingDataFunctions import WellOutputDenormalizer, GetMinMax
FLOAT_TYPE = 'float32'
tf.compat.v1.enable_eager_execution() 



class ReservoirSimulator_Proxy:

    def __init__(self, sim_inputs):

        self.initialize_variables(sim_inputs)

    def initialize_variables(self, sim_inputs) -> None:
       
        res_param = copy.deepcopy(sim_inputs)

        self.model_weights = res_param["model_weight_files"]
        self.NE = res_param["num_realizations"]
        self.Nx = res_param["model_dimensions"][0]
        self.Ny = res_param["model_dimensions"][1]
        self.Nz = res_param["model_dimensions"][2]
        self.num_prod = res_param["num_prod"] 
        self.num_inj = res_param["num_inj"]  
        self.num_wells = self.num_inj + self.num_prod
        self.control_step_size = res_param["control_step_size"] 
        self.num_control_steps = res_param["num_control_steps"]
        input_shape = (self.num_control_steps, self.NE, self.num_wells)  # (timesteps, realizations, well features)
        output_size = self.num_inj + 2*self.num_prod          # Dimension of well outputs

        ## Define LSTM Model 
        lstm_cell_units = 1024
        dense_layer = 512
        self.DL_Model = Sequential([
                                    Input(shape=input_shape),  # Input layer
                                    keras.layers.Reshape((-1, self.NE * self.num_wells)),  # Flatten realizations and features
                                    #tf.compat.v1.keras.layers.CuDNNLSTM(lstm_cell_units, return_sequences=True),
                                    LSTM(lstm_cell_units, return_sequences=True),
                                    Dense(dense_layer, activation='relu'),         # Intermediate dense layer
                                    Dense(self.NE * output_size),               # Fully connected layer for output
                                    PReLU(),
                                    keras.layers.Reshape((self.num_control_steps, self.NE, output_size))  # Reshape back to required output shape
                                ])
        self.DL_Model.load_weights(self.model_weights[0])

        
        #training/validation realization indices
        self.train_realz = res_param["training_realizations"] ## This is a LIST of lists of cluster indices excluding centroids indices
        self.val_realz = res_param["validation_realizations"] 
        self.train = res_param["train"]  #flag for training or validation. If True, data is obtained from training realization. If False, validation realizations data is used
        
        ## Decide from which cluster to choose training/validation realization 

        self.num_cores_for_training = res_param["num_cores"]
        self.num_clusters = len(self.train_realz)  ## length of LIST of lists is the number of clusters
        self.num_cores_per_cluster = self.num_cores_for_training//self.num_clusters
        self.worker_index = res_param["worker_index"]

        self.cluster_id = (self.worker_index - 1) // self.num_cores_per_cluster
        ## Ensure cluster_id is within bounds
        self.cluster_id = min(self.cluster_id, self.num_clusters - 1)
        # Randomly assign a realization ID from the determined cluster
        self.realzID = random.choice(self.train_realz[self.cluster_id])

        if self.train:
            self.realz = self.realzID #get the actual index of the training realization assigned to worker
        else:
            self.realz = self.val_realz[self.worker_index - 1] #get the actual index of the validation realization assigned to worker

        
        self.TrainFolder =  res_param["Train_Folder"]

        #set well and control bounds   
        self.water_inj_bounds = res_param["water_inj_bounds"] 
        self.prod_bhp_bounds = res_param["prod_bhp_bounds"] 
        self.wells_lower_bound = res_param["wells_lower_bound"] 
        self.wells_upper_bound = res_param["wells_upper_bound"] 
        self.reward_scaler = res_param["reward_scaler"]

        #set timing  
        self.total_days = res_param["total_days"]
        self.control_step_id = 0
        self.time_elapsed = 0
        self.num_run_per_control_step = 1  #1 because we are using a proxy here and each control step requires just 1 forward leap to get to the next unlike HFS than may be divided into several time steps
        self.num_datapoints_in_btw_control_steps = res_param["num_datapoints_in_btw_control_steps"]

        #economic parameters
        self.economic_param = {"co":  res_param["oil_price"],
                                "cw": res_param["water_prod_cost"],
                                "cwi":res_param["water_inj_cost"],
                                 "b": res_param["discount_factor"]} 
        
        # Initialization:
        self.current_well_controls = np.array([])
        self.all_well_controls = []
        self.yt1_seq = []
        self.zt_list = []
        self.FWIR_list = []
        self.NPVs = []  #cummulative NPV until the recent control step
        self.well_ctrls = np.zeros((1, self.num_control_steps, self.NE, self.num_wells))

        return None
    

    ## This method is specific to running one step in the Deep Reinforcement Learning reservoir environment
    def run_one_control_step(self, well_controls_) -> None:
        print(f"RUNNING REALIZATION #{self.realz + 1}")

        self.current_well_controls = well_controls_.copy()
        self.all_well_controls.append(self.current_well_controls)
        U_test = tf.Variable(tf.convert_to_tensor(self.current_well_controls, dtype=FLOAT_TYPE))

        res = self.Get_Well_Outputs(U_test)
        yt1 = res["yt1"]

        self.control_step_id += 1
        self.time_elapsed = self.control_step_size * self.control_step_id
        yt1 = yt1
        
        self.yt1_seq.append(tf.expand_dims(yt1, axis=1))

        return None
    
    #@tf.function
    def Get_Well_Outputs(self, U_test) -> None:
        
        t0 = time.time()
        ## Reshape well controls for use in proxy:    
        U_test_reshaped = tf.tile(tf.reshape(U_test, (1, 1, self.num_wells)), tf.constant([1, self.NE, 1]))
        self.well_ctrls[:, self.control_step_id, :, :] = U_test_reshaped  ## add controls till the latest control step..use this in LSTM proxy.
        #print('Predicting well-outputs for control step #' + str(self.control_step_id + 1))
        results = {} 
        well_outputs = self.DL_Model.predict(self.well_ctrls, batch_size = 1)  #predicts outputs for all timesteps  ==> (1, num_time_steps, NE, num_well_outputs)
        yt1 = well_outputs[:, self.control_step_id, ...] ## select welloutput for the desired control step
        results["yt1"] = yt1        
        t1 = time.time()
        print(('Time elapsed: {time: .4f} sec').format(time=abs(t1-t0)))

        return results


    def get_npv(self):
        ##calculate npv from the beginning to the latest time
        npv_till_latest_time = self.CalculateNPV(self.current_well_controls) 
        self.NPVs.append(npv_till_latest_time)  
        if self.control_step_id - 1 == 0:
            npv = npv_till_latest_time
        else:
            npv = npv_till_latest_time - self.NPVs[self.control_step_id - 2]
        return npv

    def CalculateNPV(self, well_controls):

        print('=='*20)
        print(f"calculating cummulative NPV for control step #{self.control_step_id}")
        yt1_reshaped = np.reshape(self.yt1_seq, (-1, self.control_step_id, self.NE, 2*self.num_prod + self.num_inj))

        yt1_denormalized,_ = WellOutputDenormalizer(yt1_reshaped, self.num_prod, TrainFolder = self.TrainFolder, dtype = FLOAT_TYPE)           
        ## oil production
        WOPR_y1 = yt1_denormalized[..., :self.num_prod].copy() 
        WOPR_y1 = np.concatenate((np.zeros((WOPR_y1.shape[0], 1, WOPR_y1.shape[2], WOPR_y1.shape[3])), WOPR_y1), axis=1)
        ## water production
        WWPR_y1 = yt1_denormalized[..., self.num_prod:2*self.num_prod].copy() 
        WWPR_y1 = np.concatenate((np.zeros((WWPR_y1.shape[0], 1, WWPR_y1.shape[2], WWPR_y1.shape[3])), WWPR_y1), axis=1)
        ## water injection
        #well_controls_reshaped = np.reshape(well_controls, (self.NE, self.num_wells))
        well_controls_reshaped = np.tile(np.reshape(well_controls, (1, self.num_wells)), (self.NE, 1))
        WWIR = well_controls_reshaped[:, :self.num_inj] * (self.water_inj_bounds[1] - self.water_inj_bounds[0]) + self.water_inj_bounds[0]
        
        ## field production and injection 
        FOPR_y1_seq = np.sum(WOPR_y1, axis=-1)
        FWPR_y1_seq = np.sum(WWPR_y1, axis=-1)
        FWIR = np.sum(WWIR, axis=-1)
        FWIR = np.reshape(FWIR, (self.NE,-1))
        self.FWIR_list.append(FWIR)
        FWIR_updated = np.reshape(self.FWIR_list, (self.NE,-1))

        TimeVector = np.linspace(0, self.time_elapsed , self.control_step_id + 1)
        n = len(TimeVector)
        DiscountPeriods = np.zeros(((n - 1),))
        for i in range(len(DiscountPeriods)):
            DiscountPeriods[i] = 1 / ((1 + self.economic_param["b"]) ** (TimeVector[i + 1] / 365))
        
        co = self.economic_param["co"]
        cw = self.economic_param["cw"]
        cwi = self.economic_param["cwi"]

        ## using actual values coming from the proxy
        NPV_pred = 0
        for t in range(n - 1):
            NPV_pred += (DiscountPeriods[t] * self.control_step_size * (co  * FOPR_y1_seq[0, t + 1, self.realz] - cw  * FWPR_y1_seq[0, t + 1, self.realz] - cwi * FWIR_updated[self.realz, t]) )  

        return NPV_pred
    
    def get_observation(self):

        t_values = np.linspace(0, 1, self.num_datapoints_in_btw_control_steps)
        yt1_seq_out = np.concatenate(self.yt1_seq, axis=1)  
        yt1_reshaped = yt1_seq_out.reshape((-1, self.control_step_id, self.NE, 2*self.num_prod + self.num_inj))
        ## process WWIR and Producer_BHP from input well controls as this entries are not obtained directly from the proxy
        if self.control_step_id == 1:
            old_well_controls = self.all_well_controls[self.control_step_id - 1]
            yt1_old_control_step = np.zeros((2*self.num_prod + self.num_inj))
        else:
            old_well_controls = self.all_well_controls[self.control_step_id - 2]
            yt1_old_control_step = yt1_reshaped[:, self.control_step_id - 2, self.realz,  :]

        new_well_controls = self.all_well_controls[self.control_step_id - 1]
        WWIR_old_control_step = old_well_controls[:self.num_inj] # injectors first
        WWIR_current_control_step = new_well_controls[:self.num_inj] # injectors first
        Producer_BHP_old_control_step = old_well_controls[self.num_inj:] # then producers
        Producer_BHP_current_control_step = new_well_controls[self.num_inj:] # then producers

        ## process WOPR, WWPR and IBHP. These are the original outputs obtained directly from the proxy    
        yt1_current_control_step = yt1_reshaped[:, self.control_step_id - 1, self.realz, :]  

        ## interpolate between old and new values
        interpolated_WWIR = np.array([(1 - t) * WWIR_old_control_step + t * WWIR_current_control_step for t in t_values])
        interpolated_BHP = np.array([(1 - t) * Producer_BHP_old_control_step + t * Producer_BHP_current_control_step for t in t_values])  
        interpolated_WOPR_WWPR_IBHP = np.array([(1 - t) * yt1_old_control_step + t * yt1_current_control_step for t in t_values])
        interpolated_WOPR_WWPR_IBHP = np.reshape(interpolated_WOPR_WWPR_IBHP, (-1, 2*self.num_prod + self.num_inj))

        obs_data = np.array([]).reshape(0, self.num_datapoints_in_btw_control_steps)
        obs_data = np.vstack((obs_data, interpolated_WWIR.T))
        obs_data = np.vstack((obs_data, interpolated_WOPR_WWPR_IBHP.T))
        obs_data = np.vstack((obs_data, interpolated_BHP.T))

        observations = obs_data.T.flatten()            
        observations = np.clip(observations, a_min=0, a_max=1)
        
        return observations
    
    def reset_variables(self) -> None:

        #choose a different realization from cluster every time an episode runs
        self.realzID = random.choice(self.train_realz[self.cluster_id])
        if self.train:
            self.realz = self.realzID 
        self.control_step_id = 0
        self.time_elapsed = 0
        self.current_well_controls = np.array([])
        self.all_well_controls = []
        self.yt1_seq = []
        self.zt_list = []
        self.NPVs = [] 
        self.FWIR_list = []
        self.well_ctrls = np.zeros((1, self.num_control_steps, self.NE, self.num_wells)) 

        return None
    

    ################################################################################
    #################################  THE END   ###################################
    ################################################################################




 