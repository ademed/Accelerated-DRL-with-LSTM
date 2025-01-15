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
Ninj = 5
Nprd = 3
NE = 5
Nx = 60
Ny = 60
Nz = 4
num_wells = Ninj + Nprd
num_control_timestep = 20
controlstep = 60
Total_life = 1200
n_test = 100
producer_upper_bound = 3800
producer_lower_bound = 3300
Injector_upper_bound = 1500
Injector_lower_bound = 800
#### Economic parameters for NPV calculation
co = 82 #USD oil price
cw = 5 #USD water treatment
cwi = 5 #USD water injection
discount_factor = 0.1
economic_param = {"co": co,
                  "cw": cw,
                  "cwi": cwi,
                  "b": discount_factor}
SEED = 3031
FLOAT_TYPE = 'float32'
epoch = 290

weights_base_folder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_OPTIMIZATION\E2CO_Lite_TimeDistributed\LiteWeights_HYBRID_FLUX_stopfluxatEpoch0_epoch300_batchsize4_trainsize450_seed' + str(SEED)

#### train/test min and max values
TrainFolder = r'C:\Users\aka6836\Desktop\Brugge_Large_Data\E2CODataset_BHP500to800_Rate20000to30000\Local Normalization\ReprocessedData'
TrainMinMax_Well = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_Dataset\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData\WellRatesMinMax.txt'
TestMinMax_Well = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\E2CO_Dataset\E2CODataset_BHP3300to3800_Rate800to1500\ReprocessedData_Test\WellRatesMinMax.txt'

##load weights for proxy
encoder_weight_file = weights_base_folder + r'\Epoch' + str(epoch) +r'\encoder.h5'
decoder_weight_file = weights_base_folder + r'\Epoch' + str(epoch) + r'\decoder.h5'
transition_weight_file = weights_base_folder + r'\Epoch' + str(epoch) + r'\transition.h5'
transition_output_weight_file = weights_base_folder + r'\Epoch' + str(epoch)+ r'\transition_output.h5'
#deterministic results based on a predefined SEED
#tf.keras.utils.set_random_seed(SEED)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

train_realizations = [0, 1, 2, 3, 4]
validation_realizations = [10, 11]

##############################################################################################################################
# Model Definition
model_weight_files = [encoder_weight_file, decoder_weight_file, transition_weight_file, transition_output_weight_file]

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
                                realzID = 1,
                                train = True,
                                local_normalized = False,
                                Well_GlobalMinMax = [TrainMinMax_Well, TestMinMax_Well],
                                reward_scaler = 1e8 ) #number of realizations to be examined


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
                                realzID = 1,
                                train = False,
                                local_normalized = False,
                                Well_GlobalMinMax = [TrainMinMax_Well, TestMinMax_Well],
                                reward_scaler = 1e8 ) #number of realizations to be examined



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
num_cpus = 5  #was 8
num_sim_iter = 20 #was 10
num_training_iter = 50000
analysis = tune.run("PPO",
        config={"env": ReservoirEnv,
                # "model": {
                #         "custom_model": "my_model",
                #         "max_seq_len": nstep,
                #         "custom_model_config": {
                #                                 "num_transformer_units": 4, #base 2  4
                #                                 "attention_dim": 128, #128
                #                                 "num_heads": 4, #base 2  4
                #                                 "memory_inference": nstep, 
                #                                 "memory_training": nstep,  
                #                                 "head_dim": 64, #64
                #                                 "position_wise_mlp_dim": 64,  #base 64              
                #                                 },
                #                 },
                "model":{"use_attention": True,
                         "max_seq_len": nstep,
                         "attention_num_transformer_units": 4,
                        "attention_dim": 128,
                        "attention_num_heads": 4,
                        "attention_head_dim": 64,
                        "attention_memory_inference": nstep,
                        "attention_memory_training": nstep,
                        "attention_position_wise_mlp_dim": 64},
                # "evaluation_interval" : 10,  # Evaluate with every `evaluation_interval` training iterations
                # "evaluation_duration": 10, #The number of episodes(or timesteps) we want to use in evaluation. evaluation_num_episodes is deprecated, use evaluation_duration
                # "evaluation_num_workers": 2,
                # "evaluation_config":{"env_config": sim_inputs_evaluation },
                "env_config": sim_inputs_training,
                "num_workers":num_cpus, #num of workers for sample collection (based on number of cpu)
                "num_cpus_for_driver":5, #num of cpu for training
                "num_gpus":0, #num of gpu for training
                "train_batch_size": num_cpus * nstep * num_sim_iter,  # Total number of steps  per iterations
                
                #"seed": SEED,
                "rollout_fragment_length": nstep * num_sim_iter, ## number of steps collected from each worker (or realization)
                "sgd_minibatch_size": 128, #base 256
                "num_sgd_iter": 15, #number of epochs per iteration
                "framework": "torch",
                "eager_tracing": True,

                "gamma": 0.9997,
                "kl_coeff": 0.2, #1e-12,
                "clip_param":0.2,
                #"grad_clip": 4,
                #"entropy_coeff": 0.01,
                "lr_schedule": [[0, 1e-4], [num_cpus * nstep * num_sim_iter * num_training_iter, 5e-6]],
                "entropy_coeff_schedule": [[0, 5e-3], [num_cpus * nstep * num_sim_iter * num_training_iter, 1e-5]],
                #"min_time_s_per_reporting": 10,       

                },
        ##local_dir = r"C:\Users\aka6836\ray_results",
        ##reuse_actors = True,
        stop={"training_iteration": num_training_iter},
        #name= "PPO_2024-12-18_11-38-25",
        checkpoint_freq = 10,
        #resume= "AUTO",  #uncomment this to resume from latest state. This continue trials that failed while running and it does not resume completed trials
        #restore= r"C:\Users\aka6836\ray_results\PPO_2024-12-18_11-38-25\PPO_ReservoirEnv_e2301_00000_0_2024-12-18_11-38-25\checkpoint_000499", #creates a new trial that starts from the specified checkpoint
        #local_dir= r"C:\Users\aka6836\ray_results"
        #max_failures=-1  #restart failed trial automatically
        
        )