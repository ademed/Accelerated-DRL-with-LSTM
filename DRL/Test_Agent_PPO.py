
import os
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Model_TD import *
from DRL.Proxy_Configuration import simulation_inputs
tf.compat.v1.enable_eager_execution() 


import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ReservoirEnvironment import *
from ray.rllib.algorithms.ppo import PPO
from ray.tune import ExperimentAnalysis
from ray.tune import register_env
import time


Ninj = 10
Nprd = 20
NE = 10
Nx = 139
Ny = 48
Nz = 9
num_wells = Ninj + Nprd
num_control_timestep = 5
controlstep = 180
Total_life = 900
n_test = 100
producer_upper_bound = 1000
producer_lower_bound = 500
Injector_upper_bound = 10000
Injector_lower_bound = 5000
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
epoch = 350

weights_base_folder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\BRUGGE_PROXY\E2CO_Lite_TimeDistributed\Local_Norm_Weights\LiteWeights_HYBRID_FLUX_stopfluxatEpoch0_epoch400_batchsize4_trainsize450_seed' + str(SEED)
LoadFolder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Brugge_Large_Data\E2CODataset_BHP500to1000_Rate5000to10000\Local Normalization\ReprocessedData'

#### train/test min and max values
TrainMinMax_Grid = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Brugge_Large_Data\E2CODataset_BHP500to1000_Rate5000to10000\Local Normalization\ReprocessedData\GridMinMax.txt'
TestMinMax_Grid = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Brugge_Large_Data\E2CODataset_BHP500to1000_Rate5000to10000\Local Normalization\ReprocessedData_Test\GridMinMax.txt'
TrainFolder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Brugge_Large_Data\E2CODataset_BHP500to1000_Rate5000to10000\Local Normalization\ReprocessedData'
TestFolder = r'C:\Users\aka6836\OneDrive - University of Tulsa\Desktop\Brugge_Large_Data\E2CODataset_BHP500to1000_Rate5000to10000\Local Normalization\ReprocessedData_Test'

##load weights for proxy
encoder_weight_file = weights_base_folder + r'\Epoch' + str(epoch) +r'\encoder.h5'
decoder_weight_file = weights_base_folder + r'\Epoch' + str(epoch) + r'\decoder.h5'
transition_weight_file = weights_base_folder + r'\Epoch' + str(epoch) + r'\transition.h5'
transition_output_weight_file = weights_base_folder + r'\Epoch' + str(epoch)+ r'\transition_output.h5'
#deterministic results based on a predefined SEED
#tf.keras.utils.set_random_seed(SEED)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

train_realizations = [0, 1, 2, 3, 4, 5, 6, 7]
validation_realizations = [8, 9]

model_weight_files = [encoder_weight_file, decoder_weight_file, transition_weight_file, transition_output_weight_file]

sim_inputs = simulation_inputs(
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
                                train = True ) #number of realizations to be examined

# Initialize Ray
#ray.init()
ray.init(ignore_reinit_error=True, log_to_driver=False)
# Load the trained agent
#path = r"C:\Users\aka6836\ray_results\PPO_2024-12-06_22-06-20" #r"C:\Users\aka6836\ray_results\PPO_2024-12-06_17-52-24" #new PC
#path = r"C:\Users\aka6836\ray_results\PPO_2024-12-07_04-51-31"  #old PC
path = r"C:\Users\aka6836\ray_results\PPO_2024-12-09_14-00-25"  ##new pc
analysis = ExperimentAnalysis(path, default_metric="evaluation/episode_reward_mean", default_mode="max")
t0 = time.time()
best_trail = analysis.get_best_trial()
t1 = time.time()
print(f"Best trial assigned in {t1-t0} sec")
print("="*20)
best_checkpoint = analysis.get_best_checkpoint(best_trail)
t2 = time.time()
print(f"Best checkpoint assigned in {t2-t1} sec")
print("="*20)
config = best_trail.config
agent = PPO(config=config)
agent.restore(best_checkpoint)
t3 = time.time()
print(f"Agent restore time is {t3-t2} sec")
print("="*20)
print(f"Total agent initialization time is  {t3-t0} sec")
print("="*20)



state = [np.zeros((1, 128), dtype=np.float32) for _ in range(4)]  #128 is attention dimension. 4 is the number of head used in the RMHA
NPVs = []
ACTIONS_all = []
n = 10
k = 0
for i in range(n):
    sim_inputs["realzID"] = i + 1
    if i >= 8:
        sim_inputs["train"] = False
        sim_inputs["realzID"] = k
        k = k + 1
    env = ReservoirEnv_for_Testing(sim_inputs)
    obs = env.reset()
    cum_reward = 0
    actions_for_realz = []
    print(f"RUNNING FOR REALIZATION {i+1}")
    while True:
        observations = {"agent": obs}
        
        # Compute actions using the policy
        action, _, _ = agent.compute_single_action(obs, state=state, explore=False)
        actions_for_realz.append(action)     

        # Step the environment with the computed action
        obs, r, done, _ = env.step(np.around(action))  # Extract the numpy array from the tuple
        cum_reward += r
        if done:
            break
    ACTIONS_all.append(actions_for_realz)
    NPVs.append(cum_reward)

print(NPVs)
print(f"Average NPV of all {n} realizations: ", np.mean(NPVs))
#print("NPV for True Model: ",  NPVs[-1])
