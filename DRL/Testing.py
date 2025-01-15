import os
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Model_TD import *
from DRL.Proxy_Configuration import simulation_inputs
from DRL.ReservoirSimulator_Proxy import ReservoirSimulator_Proxy

from E2CO.ProcessingDataFunctions import StateLoader, ControlLoader2
tf.compat.v1.enable_eager_execution() 
Ninj = 10
Nprd = 20
NE = 105
Nx = 139
Ny = 48
Nz = 9
num_wells = Ninj + Nprd
num_control_timestep = 5
controlstep = 180
Total_life = 900
n_test = 100
producer_upper_bound = 725
producer_lower_bound = 500
Injector_upper_bound = 30000
Injector_lower_bound = 20000
#### Economic parameters for NPV calculation
co = 63 #USD oil price
cw = 5 #USD water treatment
cwi = 5 #USD water injection
discount_factor = 0.1
economic_param = {"co": co,
                  "cw": cw,
                  "cwi": cwi,
                  "b": discount_factor}
SEED = 3031
FLOAT_TYPE = 'float32'
epoch = 210

weight_path = r'C:\Users\aka6836\OneDrive - University of Tulsa\LSTM_Proxy\LSTM_seed_3031_PRELU'
weight_file = weight_path + '\\' + r"model_weights_epoch_" + str(epoch) + '.h5'

#### train/test min and max values
TrainFolder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Sequence_Models_Brugge\Brugge_Data_All_Realizations\ReprocessedData'


num_clusters = 3
###  clustered realizations
filepath_for_cluster_labels = r"C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Sequence_Models_Brugge\cluster_labels_for_all_realz.pkl"
with open(filepath_for_cluster_labels, 'rb') as f1:
        cluster_labels = pickle.load(f1)

centroid_indices = [23, 67, 5]
clusters_indices_excluding_centroids = []
for k in range(num_clusters):
        cluster_indices = np.where(cluster_labels == k)[1]  ## get indices of the k-th cluster
        cluster_indices = [j for j in cluster_indices if j != centroid_indices[k]] ##exlude centroid index from the cluster indices
        clusters_indices_excluding_centroids.append(cluster_indices)

train_realizations = clusters_indices_excluding_centroids
validation_realizations = centroid_indices   #centroids

num_cores = 19  #number of processors for data gathering.. 1 per training realization
num_cores_per_cluster = num_cores//num_clusters


##############################################################################################################################
# Model Definition
model_weight_files = [weight_file]

sim_inputs_training = simulation_inputs(
                                model_weight_files = model_weight_files,
                                Train_Folder = TrainFolder, # train folder for denormalizing well output predictions
                                num_prod = Nprd, #number of producers
                                num_inj = Ninj, #number of injectors
                                model_dimensions = [Nx, Ny, Nz],
                                water_inj_bounds = [Injector_lower_bound, Injector_upper_bound], #lower and upper injection rate bound for all injectors
                                prod_bhp_bounds = [producer_lower_bound, producer_upper_bound],  #upper and lower BHP bound for all producers
                                control_step_size = controlstep, #control step size
                                num_control_steps = num_control_timestep,  #number of control steps
                                economic_parameters = economic_param, #parameters for calculating NPV
                                num_realizations = NE,
                                train_realz= train_realizations,
                                val_realz = validation_realizations,
                                num_datapoints_in_btw_control_steps = 11,
                                worker_index = 3,
                                train = False,
                                num_cores = num_cores, ## number of cores of data gathering
                                reward_scaler = 1e9 ) #number of realizations to be examined


LoadFolder = r'C:\\Users\\aka6836\\OneDrive - University of Tulsa\\Desktop\\Sequence_Models_Brugge\\Brugge_Data_All_Realizations\\ReprocessedData'
u_test = ControlLoader2(LoadFolder=LoadFolder+"_Test", filename='u_test.pkl', dtype=FLOAT_TYPE)
well_controls = u_test[0,:,0,:]
res_sim = ReservoirSimulator_Proxy(sim_inputs_training)


NPVs = []
import time
t0 = time.time()
for _ in range(10):
    for i in range(5):
        res_sim.run_one_control_step(well_controls[i,...])
        npv = res_sim.get_npv()
        obs = res_sim.get_observation()
        print(obs.shape)
        NPVs.append(npv)
    print("final NPV:" , res_sim.NPVs[-1])
    res_sim.reset_variables()
t1 = time.time()
print(f"TOTAL RUN TIME: {t1-t0}")  #651.21 sec 
print("==="*30)
#print("cummulative NPV: ",np.array(res_sim.NPVs.remote())/1e9)
print("==="*30)
print("Immediate NPV: ", np.array(NPVs)/1e9)

