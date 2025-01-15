import numpy as np



def simulation_inputs(
                      model_weight_files,
                      Train_Folder,
                      num_prod, #number of producers
                      num_inj, #number of injectors
                      model_dimensions,
                      water_inj_bounds, #upper and lower bound for all injectors
                      prod_bhp_bounds,  #upper and lower BHP bound for all producers
                      control_step_size, #control step size
                      num_control_steps,  #number of control steps
                      economic_parameters, #parameters for calculating NPV
                      num_realizations,
                      train_realz,
                      val_realz,
                      num_datapoints_in_btw_control_steps,
                      worker_index,
                      train,
                      num_cores = None,
                      reward_scaler = 1e9        
                        ):
    sim_inputs = {}
    sim_inputs["worker_index"] = worker_index
    sim_inputs["num_datapoints_in_btw_control_steps"] = num_datapoints_in_btw_control_steps
    #set DL proxy
    sim_inputs["state_init"] = None
    sim_inputs["model_weight_files"] = model_weight_files
    sim_inputs["Train_Folder"] = Train_Folder
    sim_inputs["model_dimensions"] = model_dimensions
    sim_inputs["training_realizations"] = train_realz
    sim_inputs["validation_realizations"] = val_realz
    sim_inputs["train"] = train
    sim_inputs["num_cores"] = num_cores
    sim_inputs["reward_scaler"] = reward_scaler

    #set well and control bounds
    sim_inputs["num_prod"] = num_prod
    sim_inputs["num_inj"]  = num_inj
    sim_inputs["water_inj_bounds"] = water_inj_bounds
    sim_inputs["prod_bhp_bounds"]  = prod_bhp_bounds
    sim_inputs["wells_lower_bound"] = np.concatenate([[water_inj_bounds[0]] * num_inj, [prod_bhp_bounds[0]] * num_prod])
    sim_inputs["wells_upper_bound"] = np.concatenate([[water_inj_bounds[1]] * num_inj, [prod_bhp_bounds[1]] * num_prod])
   
    #set timing

    sim_inputs["control_step_size"] = control_step_size
    sim_inputs["num_control_steps"] = num_control_steps
    sim_inputs["total_days"] = control_step_size * num_control_steps

    #economic parameters
    sim_inputs["oil_price"] = economic_parameters["co"]
    sim_inputs["water_prod_cost"] = economic_parameters["cw"]
    sim_inputs["water_inj_cost"] = economic_parameters["cwi"]
    sim_inputs["discount_factor"] = economic_parameters["b"]

    sim_inputs["num_realizations"] = num_realizations

    return sim_inputs