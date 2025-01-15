import os
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Model_TD import *
from DRL.Proxy_Configuration import simulation_inputs

from E2CO.ProcessingDataFunctions import StateLoader
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

num_cores = 16#21  #number of processors for data gathering.. 1 per training realization #16 for old
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
                                worker_index = 1,
                                train = True,
                                num_cores = num_cores, ## number of cores of data gathering
                                reward_scaler = 1e9 ) #number of realizations to be examined


sim_inputs_evaluation = simulation_inputs(
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
                                worker_index = 1,
                                train = False,
                                num_cores = num_cores, ## number of cores of data gathering
                                reward_scaler = 1e9 ) #number of realizations to be examined



###############################################################################################
################################ SET UP DRL TRAINING  #########################################
###############################################################################################
import ray
from ray import tune
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from DRL.ReservoirEnvironment import *
#from DRL.network_model_attention_cnn import GTrXLNet as MyModel
from DRL.GTRXL_model_TORCH import GTrXLNet as MyModel

ray.init(ignore_reinit_error=True, log_to_driver=False)


ModelCatalog.register_custom_model("my_model", MyModel)

nstep = num_control_timestep
num_cpus = num_cores  #was 8
num_sim_iter = 30 #30 #was 10
num_training_iter = 30000
analysis = tune.run("PPO",
        config={"env": ReservoirEnv,
                "model":{"use_attention": True,
                         "max_seq_len": nstep,
                         "attention_num_transformer_units": 4, #4
                        "attention_dim": 128, #128
                        "attention_num_heads": 4, #4
                        "attention_head_dim": 64, #64
                        "attention_memory_inference": nstep,
                        "attention_memory_training": nstep,
                        "attention_position_wise_mlp_dim": 64}, #64
                "evaluation_interval" : 10,  # Evaluate with every `evaluation_interval` training iterations
                "evaluation_duration": 10, #The number of episodes(or timesteps) we want to use in evaluation. evaluation_num_episodes is deprecated, use evaluation_duration
                "evaluation_num_workers": num_clusters,
                "evaluation_config":{"env_config": sim_inputs_evaluation },
                "env_config": sim_inputs_training,
                "num_workers":num_cpus, #num of workers for sample collection (based on number of cpu)
                "num_cpus_for_driver":4, #num of cpu for training
                "num_gpus":0, #num of gpu for training
                "train_batch_size": num_cpus * nstep * num_sim_iter,  # Total number of steps  per iterations
                
                #"seed": SEED,
                "rollout_fragment_length": nstep * num_sim_iter, ## number of steps collected from each worker (or realization)
                "sgd_minibatch_size": 256, #base 256
                "num_sgd_iter": 15, #number of epochs per iteration
                "framework": "torch",
                "eager_tracing": True,

                "gamma": 0.9997,
                #"kl_coeff": 0.8, #1e-12,
                #"clip_param":0.2,
                #"grad_clip": 4,
                #"entropy_coeff": 0.01,
                "lr_schedule": [[0, 1e-4], [num_cpus * nstep * num_sim_iter * num_training_iter, 5e-6]],
                "entropy_coeff_schedule": [[0, 5e-4], [num_cpus * nstep * num_sim_iter * num_training_iter, 1e-6]],
                #"min_time_s_per_reporting": 10,       

                },
        ##local_dir = r"C:\Users\aka6836\ray_results",
        ##reuse_actors = True,
        stop={"training_iteration": num_training_iter},
        name= "PPO_2025-01-10_04-39-35",
        checkpoint_freq = 10,
        #resume= "AUTO",  #uncomment this to resume from latest state. This continue trials that failed while running and it does not resume completed trials
        restore= r"C:\Users\aka6836\ray_results\PPO_2025-01-10_04-39-35\PPO_ReservoirEnv_2f611_00000_0_2025-01-10_04-39-35\checkpoint_000999", #creates a new trial that starts from the specified checkpoint
        #local_dir= r"C:\Users\aka6836\ray_results"
        #max_failures=-1  #restart failed trial automatically
        
        )