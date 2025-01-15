import numpy as np
def simulation_inputs(num_prod, #number of producers
                     num_inj, #number of injectors
                     water_inj_bounds, #upper and lower bound for all injectors
                     prod_bhp_bounds,  #upper and lower BHP bound for all producers
                     time_increment,   #time stepping used in schedule_python.ixf
                     control_step_size, #control step size
                     num_control_steps,  #number of control steps
                     economic_parameters, #parameters for calculating NPV
                     group_name,  #group name for wells in reservoir. needed for the schedule file generation
                     num_processors,
                     restart, #bool to signify as restart or not
                     directory,
                     num_realizations,
                     realzID
                        ):
    sim_inputs = {}

    #set well and control bounds
    sim_inputs["num_prod"] = num_prod
    sim_inputs["num_inj"]  = num_inj
    sim_inputs["water_inj_bounds"] = water_inj_bounds
    sim_inputs["prod_bhp_bounds"]  = prod_bhp_bounds
    sim_inputs["wells_lower_bound"] = np.concatenate([[water_inj_bounds[0]] * num_inj, [prod_bhp_bounds[0]] * num_prod])
    sim_inputs["wells_upper_bound"] = np.concatenate([[water_inj_bounds[1]] * num_inj, [prod_bhp_bounds[1]] * num_prod])
   
    #set timing
    sim_inputs["time_increment"] = time_increment
    sim_inputs["control_step_size"] = control_step_size
    sim_inputs["num_stepping_iter"] = int(control_step_size / time_increment)
    sim_inputs["num_control_steps"] = num_control_steps
    sim_inputs["total_days"] = control_step_size * num_control_steps

    #economic parameters
    sim_inputs["oil_price"] = economic_parameters["co"]
    sim_inputs["water_prod_cost"] = economic_parameters["cw"]
    sim_inputs["water_inj_cost"] = economic_parameters["cwi"]
    sim_inputs["discount_factor"] = economic_parameters["b"]
    sim_inputs["group_name"] = group_name

    sim_inputs["directory"] = directory
    sim_inputs["num_processors"] =num_processors
    sim_inputs["restart"] = restart

    sim_inputs["num_realizations"] = num_realizations
    sim_inputs["realzID"] = realzID


    return sim_inputs

