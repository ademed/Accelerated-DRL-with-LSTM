import sys, os
sys.path.append(os.getcwd())
from DRL.ReservoirEnvironment import ReservoirEnv_for_HFS
from DRL.environment_configuration import simulation_inputs
from DRL.Utilities import delete_files_2
import numpy as np

import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO
from ray.tune import ExperimentAnalysis
from ray.tune import register_env
import time

##SETUP

#set well and control bounds
num_prod = 20
num_inj = 10
water_inj_bounds = [20000, 30000]
prod_bhp_bounds = [500, 800]

#set timing
time_increment = 36
control_step_size = 180
num_stepping_iter = int(control_step_size/time_increment)
num_control_steps = 5
num_processors = 12
restart = False

group_name = "G1"

economic_param = {"co": 63,
                  "cw": 5,
                  "cwi": 5,
                  "b": 0.1}
#directory for the gsg file and other simulation files that need to be copied for successfully running the simulation
directory = r"C:\Users\aka6836\OneDrive - University of Tulsa\DRL_Proxy_Enabled - V3\GSG"  
directory2 = r"C:\Users\aka6836\OneDrive - University of Tulsa\DRL_Proxy_Enabled - V3\GSG\Evaluation"
sim_inputs = simulation_inputs(num_prod = num_prod,
                               num_inj = num_inj,
                               water_inj_bounds = water_inj_bounds,
                               prod_bhp_bounds = prod_bhp_bounds,
                               time_increment = time_increment,
                               control_step_size = control_step_size,
                               num_control_steps = num_control_steps,
                               economic_parameters = economic_param,
                               group_name = group_name,
                               num_processors = num_processors,
                               restart = restart,
                               directory = directory,
                               num_realizations = 1,
                               realzID= 1)

# Load the trained agent
path = r"C:\Users\aka6836\ray_results\PPO_2024-12-17_13-35-22"
analysis = ExperimentAnalysis(path, default_metric="evaluation/episode_reward_mean", default_mode="max")
best_trail = analysis.get_best_trial()
best_checkpoint = analysis.get_best_checkpoint(best_trail)
config = best_trail.config
agent = PPO(config=config)
agent.restore(best_checkpoint)




########################################################
############## ALL MODELS #############################
########################################################
state = [np.zeros((1, 128), dtype=np.float32) for _ in range(4)]
ACTIONS_all = []
NPVs = []
for i in range(11):
    sim_inputs["realzID"] = i + 1
    env = ReservoirEnv_for_HFS(sim_inputs)
    obs = env.reset()
    cum_reward = 0
    actions_for_realz = []
    print(f"RUNNING FOR REALIZATION {i+1}")
    while True:
        observations = {"agent": obs}
        
        # Compute actions using the policy
        action, _, _ = agent.compute_single_action(obs, state=state, explore = True)
        actions_for_realz.append(action)     

        # Step the environment with the computed action
        obs, r, done, _ = env.step(np.around(action))  # Extract the numpy array from the tuple
        cum_reward += r
        if done:
            break
    ACTIONS_all.append(actions_for_realz)
    NPVs.append(cum_reward)

print(NPVs)
print("Average NPV of all 10 realizations: ", np.mean(NPVs[:10]))
print("NPV for True Model: ",  NPVs[-1])


