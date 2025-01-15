import sys, os

sys.path.append(os.getcwd())
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from DRL.ReservoirSimulator_Proxy import ReservoirSimulator_Proxy
from DRL.ReservoirSimulator_HFS import ReservoirSimulator_HFS
from ray.rllib.evaluation.worker_set import WorkerSet

from DRL.ReservoirSimulator_Proxy import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Layers_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Loss_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Blocks_TD import *
from E2CO_Lite_TimeDistributed.E2CO_Lite_Model_TD import *

FLOAT_TYPE = 'float32'

class ReservoirEnv(gym.Env):

    def __init__(self, env_config):
        """
        Must Define self.observation_space and self.action_space here
        """

        self.sim_input = env_config
        
        self.sim_input["worker_index"] = env_config.worker_index

        #start simulation_proxy
        self.res_sim = ReservoirSimulator_Proxy(self.sim_input)
        #setup action and observation space
        self.setup_spaces()
        #cummulative reward
        self.cum_reward = 0
        

    def setup_spaces(self):
        #initialize action space 
        action_low = np.array([0.0]*self.res_sim.num_wells) 
        action_high = np.array([1.0]*self.res_sim.num_wells) 
        self.action_space = Box(low = action_low , high = action_high, dtype=np.float32) #, shape=np.array((self.res_sim.num_wells,)))

        #initialize observation space
        self.num_obs_data = (3 * self.res_sim.num_prod + 2 * self.res_sim.num_inj) * self.res_sim.num_datapoints_in_btw_control_steps
        obs_low = np.array([0.0] * self.num_obs_data) 
        obs_high = np.array([1.0] * self.num_obs_data) 
        self.observation_space = Box(low = obs_low, high = obs_high, dtype=np.float32)# shape=np.array((self.num_obs_data,)))

    def reset(self):
        """
        Returns: the observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """     
        self.res_sim.reset_variables()
        self.cum_reward = 0
        observation  = np.array([0.0] * self.num_obs_data)   #control started from the very beginning

        return np.array(observation, dtype=np.float32)
        

    def step(self, action):
        """
        Returns: the next observation, the reward, done and optionally additional info
        """
        #check that current action is in action space
        assert self.action_space.contains(action)
         
        #run one control step
        self.res_sim.run_one_control_step(action)
        
        #get observation
        observation = self.res_sim.get_observation()
       
        #get reward
        reward = self.res_sim.get_npv()/self.res_sim.reward_scaler
        self.cum_reward = self.res_sim.NPVs[self.res_sim.control_step_id - 1]/self.res_sim.reward_scaler
        print(f"reward at control step #{self.res_sim.control_step_id }:",reward)
        print(f"cummulative reward at control step #{self.res_sim.control_step_id}:",self.cum_reward)
        terminated = False
        terminated = self.res_sim.control_step_id  >= self.res_sim.num_control_steps
    

        return observation, reward, terminated, {}

    def render(self):
        """
        Returns: None
        Shows the current environment state e.g. a graphical window in 'cartpole-v1
        This method must be implemented, but it is ok to have an empty implementation if rendering is not important
        """
        pass
     
    def close(self):
        """
        Returns: None
        this method is optional. Used to cleanup resources 
        """
        pass
        
    def seed(self, seed = None):
        """
        Returns: List of seeds
        This method is optional. used to set seeds for the environment's random number generator for obtaining deterministic behaviour
        """
        return




class ReservoirEnv_for_Testing(gym.Env):

    def __init__(self, env_config):
        """
        Must Define self.observation_space and self.action_space here
        """

        self.sim_input = env_config
        

        #self.sim_input["realzID"] = env_config.worker_index

        #start simulation_proxy
        self.res_sim = ReservoirSimulator_Proxy(self.sim_input)
        self.train = self.sim_input["train"] 
        #setup action and observation space
        self.setup_spaces()
        #cummulative reward
        self.cum_reward = 0
        

    def setup_spaces(self):
        #initialize action space 
        action_low = np.array([0.0]*self.res_sim.num_wells) 
        action_high = np.array([1.0]*self.res_sim.num_wells) 
        self.action_space = Box(low = action_low , high = action_high, dtype=np.float32) #, shape=np.array((self.res_sim.num_wells,)))

        #initialize observation space
        self.num_obs_data = (3 * self.res_sim.num_prod + 2 * self.res_sim.num_inj) * self.res_sim.num_datapoints_in_btw_control_steps
        obs_low = np.array([0.0] * self.num_obs_data) 
        obs_high = np.array([1.0] * self.num_obs_data) 
        self.observation_space = Box(low = obs_low, high = obs_high, dtype=np.float32)# shape=np.array((self.num_obs_data,)))

    def reset(self):
        """
        Returns: the observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """     
        self.res_sim.reset_variables()
        self.cum_reward = 0
        observation  = np.array([0.0] * self.num_obs_data)   #control started from the very beginning

        return np.array(observation, dtype=np.float32)
        

    def step(self, action):
        """
        Returns: the next observation, the reward, done and optionally additional info
        """
        #check that current action is in action space
        #assert self.action_space.contains(action)
        if self.action_space.contains(action):
            print(f"Max Action: { np.max(action)} and Min Action: { np.min(action)} ")
            print("GOOD ACTION = ", action)
            
            well_controls = action
        else:
            print(f"Max Action: { np.max(action)} and Min Action: { np.min(action)} ")
            print("BAD ACTION: ", action)
            
            #well_controls = np.clip(action, a_min = 0, a_max=1)
        #set well controls
        #well_controls = action
        
        #run one control step
        self.res_sim.run_one_control_step(well_controls)
        
        #get observation
        observation = self.res_sim.get_observation()
       
        #get reward
        reward = self.res_sim.get_npv()/1e9
        self.cum_reward = self.res_sim.NPVs[self.res_sim.control_step_id - 1]/1e9
        print(f"reward at control step #{self.res_sim.control_step_id }:",reward)
        print(f"cummulative reward at control step #{self.res_sim.control_step_id}:",self.cum_reward)
        terminated = False
        terminated = self.res_sim.control_step_id  >= self.res_sim.num_control_steps
    

        return observation, reward, terminated, {}

    def render(self):
        """
        Returns: None
        Shows the current environment state e.g. a graphical window in 'cartpole-v1
        This method must be implemented, but it is ok to have an empty implementation if rendering is not important
        """
        pass
     
    def close(self):
        """
        Returns: None
        this method is optional. Used to cleanup resources 
        """
        pass
        
    def seed(self, seed = None):
        """
        Returns: List of seeds
        This method is optional. used to set seeds for the environment's random number generator for obtaining deterministic behaviour
        """
        return
    
class ReservoirEnv_for_HFS(gym.Env):

    def __init__(self, env_config):
        """
        Must Define self.observation_space and self.action_space here
        """

        self.sim_input = env_config


        #start simulator
        self.res_sim = ReservoirSimulator_HFS(self.sim_input)
        #setup action and observation space
        self.setup_spaces()
        #cummulative reward
        self.cum_reward = 0
        

    def setup_spaces(self):
        #initialize action space 
        action_low = np.array([0.0]*self.res_sim.num_wells)
        action_high = np.array([1.0]*self.res_sim.num_wells)
        self.action_space = Box(low = action_low , high = action_high, dtype=np.float32) #, shape=np.array((self.res_sim.num_wells,)))

        #initialize observation space
        self.num_obs_data = (3 * self.res_sim.num_prod + 2 * self.res_sim.num_inj) * self.res_sim.num_run_per_control_step
        obs_low = np.array([0.0] * self.num_obs_data) 
        obs_high = np.array([1.0] * self.num_obs_data) 
        self.observation_space = Box(low = obs_low, high = obs_high, dtype=np.float32)# shape=np.array((self.num_obs_data,)))

    def reset(self):
        """
        Returns: the observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """     
        self.res_sim.reset_variables()
        self.cum_reward = 0
        observation  = np.array([0.0]*self.num_obs_data)  #control started from the very beginning

        return np.array(observation, dtype=np.float32)
        

    def step(self, action):
        """
        Returns: the next observation, the reward, done and optionally additional info
        """
        #check that current action is in action space
        assert self.action_space.contains(action)

        #set well controls
        well_controls = action * (self.sim_input["wells_upper_bound"] - self.sim_input["wells_lower_bound"]) + self.sim_input["wells_lower_bound"]
        
        #run one control step
        self.res_sim.run_one_control_step(well_controls)
        
        #get observation
        observation = self.res_sim.get_observation()
       
        #get reward
        reward = self.res_sim.get_npv()/1e8
        self.cum_reward = self.res_sim.NPVs[self.res_sim.control_step_id - 1]/1e8
        print(f"reward at control step #{self.res_sim.control_step_id }:",reward)
        print(f"cummulative reward at control step #{self.res_sim.control_step_id}:",self.cum_reward)


        terminated = False
        terminated = self.res_sim.control_step_id  >= self.res_sim.num_control_steps
    

        return observation, reward, terminated, {}

    def render(self):
        """
        Returns: None
        Shows the current environment state e.g. a graphical window in 'cartpole-v1
        This method must be implemented, but it is ok to have an empty implementation if rendering is not important
        """
        pass
     
    def close(self):
        """
        Returns: None
        this method is optional. Used to cleanup resources 
        """
        pass
        
    def seed(self, seed = None):
        """
        Returns: List of seeds
        This method is optional. used to set seeds for the environment's random number generator for obtaining deterministic behaviour
        """
        return