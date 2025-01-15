import csv
import numpy as np
import subprocess
import os


def run_simulation(batch_file_path, control_step_id, platform = "subprocess"):
    if platform == "subprocess":
        run_simulation_using_subprocess(batch_file_path, control_step_id)
    elif platform == "os":
        run_simulation_using_os(batch_file_path, control_step_id)

def run_simulation_using_subprocess(batch_file_path,control_step_id):
    ##Execute the batch file using subprocess
    ##Full path to the batch file
    b_file_path = r"" + batch_file_path
    full_batch_file_path = os.path.join(os.getcwd(), b_file_path)
  
    # Run the batch file using subprocess and capture output/error
    try:
        result = subprocess.run(full_batch_file_path, capture_output=True, text=False)
        if result.returncode != 0:
            print("Output from Subprocess:", "Simulation Failed!!")
        else:
            print(f"Simulation executed successfully for control step #{control_step_id}.")
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Return Code:", e.returncode)


def run_simulation_using_os(batch_file_path,control_step_id):
    import os
    # Execute the batch file using os.system()
    try:
        os.system(batch_file_path)
        print(f"Simulation executed successfully for control step #{control_step_id}.")
    except Exception as e:
        print(f"Error: {e}")



#ERASE CCONTENT FROM ANY FILE
def erase(filename):
    # Open the file in write mode, which clears existing content
    with open(filename, 'w') as f:
        pass  # This effectively clears the file by not writing anything

###############################################
##### ---SCHEDULE.IXF GENERATION #####
##############################################
##START SCHEDULE.IXF  
def schedule_SOF(num_inj, num_prod, group_name, filename):
    members = []
    # Generate "I" members
    for i in range(1, num_inj + 1):
        members.append(f'"I{i}"')

    # Generate "P" members
    for i in range(1, num_prod + 1):
        members.append(f'"P{i}"')

    # Join the members list into a string
    members_string = " ".join(members)

    # Construct the group string
    group_string = f'Group "{group_name}" {{\n'
    group_string += f'    Members=[ Well( {members_string} ) ]\n'
    group_string += "}"
    with open(filename, "w") as _file:
        _file.write("MODEL_DEFINITION\n\n\n\n")
        _file.write("START\n\n")
        _file.write(group_string+ '\n\n')

##APPEND END OF FILE FOR SCHEDULE.IXF
def append_EOF(filename):
    with open(filename, "a") as _file:
        _file.write("\n\n\nEND_INPUT")

##APPEND TIME FOR SCHEDULE.IXF
def append_Time(num_stepping_iter,time_increment,time_elapsed, filename):
    time_stepping_string = ""
    for _ in range(num_stepping_iter):
        time_elapsed += time_increment
        time_stepping_string += f'TIME {float(time_elapsed)}\n'

    with open(filename, "a") as _file:
        _file.write(time_stepping_string)
    return time_elapsed

##APPEND WELL CONTROLS FOR SCHEDULE.IXF AT SPECIFIED CONTROL STEP
def append_ControlStep(control_step_id, well_controls, num_inj, num_prod, filename):
    #APPEND CONTROL STEPS
    control_step_string = "################################################\n"
    control_step_string += f"## CONTROL STEP #{control_step_id}: \n"
    control_step_string += "################################################\n\n"
    #Injectors
    if control_step_id == 1:
        for i in range(1, num_inj + 1):
            control_step_string += f'Well "I{i}" {{\n'
            control_step_string += "    Status=OPEN\n"
            control_step_string += "    Type=WATER_INJECTOR\n"
            control_step_string += f"    Constraints=[ Constraint({well_controls[i-1]} WATER_INJECTION_RATE)]\n"
            control_step_string += "    HonorInjectionStreamAvailability = FALSE\n"
            control_step_string += "}\n\n"
        #Producers
        prod_index= num_inj #start of index for producers
        for i in range(1, num_prod + 1):
            control_step_string += f'Well "P{i}" {{\n'
            control_step_string += "    Status=OPEN\n"
            control_step_string += "    Type=PRODUCER\n"
            control_step_string += f"    Constraints=[ Constraint({well_controls[prod_index]} BOTTOM_HOLE_PRESSURE)]\n"
            control_step_string += "}\n\n"
            prod_index += 1
    else:
        for i in range(1, num_inj + 1):
            control_step_string += f'Well "I{i}" {{\n'
            control_step_string += "    Status=OPEN\n"
            control_step_string += "    Type=WATER_INJECTOR\n"
            control_step_string += "    remove_all_constraints( )\n"
            control_step_string += f"    Constraints=[ Constraint({well_controls[i-1]} WATER_INJECTION_RATE)]\n"
            control_step_string += "    HonorInjectionStreamAvailability = FALSE\n"
            control_step_string += "}\n\n"
        #Producers
        prod_index= num_inj #start of index for producers
        for i in range(1, num_prod + 1):
            control_step_string += f'Well "P{i}" {{\n'
            control_step_string += "    Status=OPEN\n"
            control_step_string += "    Type=PRODUCER\n"
            control_step_string += "    remove_all_constraints( )\n"
            control_step_string += f"    Constraints=[ Constraint({well_controls[prod_index]} BOTTOM_HOLE_PRESSURE)]\n"
            control_step_string += "}\n\n"
            prod_index += 1

    with open(filename, "a") as _file:
        _file.write(control_step_string)

##ERASE EOF FROM SCHEDULE.IXF FILE
def erase_EOF(filename):
    # Read the content of the file
    with open(filename, "r") as file:
        lines = file.readlines()

    # Determine where the portion you want to erase starts
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "END_INPUT":
            start_index = i
            break

    # Rewrite the file, excluding that portion
    with open(filename, "w") as file:
        for line in lines[:start_index]:
            file.write(line)


###############################################
##### ---FM_EDIT.IXF GENERATION #####
##############################################

##START FM_EDIT FILE         
def fm_edits_SOF(filename):
    with open(filename, "w") as _file:
        _file.write("MODEL_DEFINITION\n\n\n\n")

##ERASE RESTART INSTRUCTION FROM FM_EDIT.IXF FILE
def erase_EOF_fm_edit(restart_string, filename):
    # Read the content of the file
    with open(filename, "r") as file:
        lines = file.readlines()

    # Determine where the portion you want to erase starts
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == restart_string:
            start_index = i
            break

    # Rewrite the file, excluding that portion
    with open(filename, "w") as file:
        for line in lines[:start_index]:
            file.write(line)

def append_restart(restart_time, ID,  fm_edit_filename):
    restart_string = f'TIME {restart_time} SAVE_RESTART "restart_{ID}_{restart_time}"'
    with open(fm_edit_filename, "a") as _file:
        _file.write(restart_string)
    
    return restart_string


###############################################
##### ---AFI_FILE.AFI GENERATION #####
##############################################

##APPEND RESTART INSTRUCTION IN .AFI_FILE
def append_restart_afi(restart_time, ID, afi_file):
    afi_restart_string = f'INCLUDE "restart_{ID}_{restart_time}" {{ time="{restart_time}" type="restart" }}'
    with open(afi_file, "a") as _file:
        _file.write(afi_restart_string)
    
    return afi_restart_string

##ERASE RESTART INSTRUCTION IN .AFI_FILE
def erase_EOF_afi(afi_restart_string, filename, isStart = False):
    # Read the content of the file
    if not isStart:
        with open(filename, "r") as file:
            lines = file.readlines()

        # Determine where the portion you want to erase starts
        start_index = 0
        for i, line in enumerate(lines):
            if line.strip() == afi_restart_string:
                start_index = i
                break

        # Rewrite the file, excluding that portion
        with open(filename, "w") as file:
            for line in lines[:start_index]:
                file.write(line)








###############################################
##### --- NPV #####
##############################################

##CALCULATE NPV WITH WELL RATES
def CalculateNPV(file_names, economic_parameters, use_simulation_timestep = False, timestep_size = None, total_days = None):
    if use_simulation_timestep:
        return CalculateNPV_with_Simulation_Timestep(file_names, economic_parameters)
    else:
        if timestep_size is None and total_days is None:
            raise ValueError("time step size and total simulation time should be specified")
        return CalculateNPV_with_UserDefined_Timestep(file_names, economic_parameters, timestep_size, total_days)

##CALCULATE NPV WITH FIELD CUMMULATIVE PRODUCTION TOTAL  
def CalculateNPV_FieldCumProd(file_name, economic_parameters, use_simulation_timestep = False, timestep_size = None, total_days = None):
    if use_simulation_timestep:
        return CalculateNPV_with_Simulation_Timestep_FieldCumProd(file_name, economic_parameters)
    else:
        if timestep_size is None and total_days is None:
            raise ValueError("time step size and total simulation time should be specified")
        return CalculateNPV_with_UserDefined_Timestep_FieldCumProd(file_name, economic_parameters, timestep_size, total_days)
    
def CalculateNPV_with_Simulation_Timestep_FieldCumProd(file_name, economic_parameters):
    #TODO implement fgpt
    fopt, fwit, fwpt, flpt, fgpt, Tn = GetFieldTotalProduction(file_name)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]
    n = len(fopt)
    #average barrel produced/injected at every timestep (multiplied by their cost)
    oilProd_revenue = (fopt[1:n] - fopt[0:n-1])*co
    waterInj_cost = (fwit[1:n] - fwit[0:n-1])*cwi
    waterProd_cost = (fwpt[1:n] - fwpt[0:n-1])*cw
    #discounted time 
    t = [1/((1 + b)**(Tn[i+1]/365)) for i in range(n-1)]
    t = np.array(t)
    #profit
    profit = oilProd_revenue - waterInj_cost - waterProd_cost
    npv = np.dot(t,profit)
    return npv 

def CalculateNPV_with_UserDefined_Timestep_FieldCumProd(file_name, economic_parameters, timestep_size, time_elapsed):
    fopt, fwit, fwpt, flpt, fgpt, time_vec = GetFieldTotalProduction(file_name)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]
    dt = timestep_size #control time step

    Total_Days = time_elapsed
    Nc = int(Total_Days/dt) #Number of control steps
    time_steps = dt*np.ones(Nc) #time steps in an array
    Tn = np.cumsum(time_steps, axis=0) #total time elapsed for all the controlsteps, e.g if we have control step size of 180 and Nc = 5, Tn contains: 180, 360, 540,720 and 900
    Tn = np.concatenate(([0.0], Tn))
    
    ## indices of the required T_n in the original time_vec.  
    ##=========I had to implement this way because the simulation timestep could change due to convergence issues, thereby resulting in more timesteps than necessary===##
    # Initialize an empty set to store unique values encountered
    seen_values = set()
    # Initialize an empty list to store indices of required T_n
    indices = []
    # Iterate over time_vec
    for i, value in enumerate(time_vec):
        # Check if the value is in Tn and not already seen
        if value in Tn and value not in seen_values:
            # Add the index to indices
            indices.append(i)
            # Add the value to seen_values
            seen_values.add(value)

    ##After the loop above, if indices is still not same length as Tn, just interpolate for the values of prod data for the required timesteps
    if len(indices) is not len(Tn):
        from scipy.interpolate import CubicSpline
        
        ##Discard repititions in time_vec and take the mean of values in welldata for which time_vec is repeated . This is done to ensure cubic spline works fine
        unique_time_vec, idx = np.unique(time_vec, return_inverse=True)
        unique_fopt = np.array([np.mean(fopt[idx == i], axis=0) for i in range(len(unique_time_vec))])
        unique_fwit = np.array([np.mean(fwit[idx == i], axis=0) for i in range(len(unique_time_vec))])
        unique_fwpt = np.array([np.mean(fwpt[idx == i], axis=0) for i in range(len(unique_time_vec))])

        #form cubic spline
        cs_fopt = CubicSpline(unique_time_vec, unique_fopt)
        cs_fwit = CubicSpline(unique_time_vec, unique_fwit)
        cs_fwpt = CubicSpline(unique_time_vec, unique_fwpt)

        #interpolate for the required Tn
        fopt_Nc = cs_fopt(Tn)
        fwit_Nc = cs_fwit(Tn)
        fwpt_Nc = cs_fwpt(Tn)
    else:

        ##I know I know...this line is superfluous but, you can never be too careful!!
        assert len(indices)==len(Tn), "size of total time and indices should be equal"
        
        #cummulative oil produced at each control steps
        fopt_Nc = np.array([fopt[i] for i in indices])

        #cummulative water injected at each control steps
        fwit_Nc = np.array([fwit[i] for i in indices])

        #cummulative water produced at each control steps
        fwpt_Nc = np.array([fwpt[i] for i in indices])

    n = len(indices)
    #Total average barrel produced/injected between two consecutive control steps (multiplied by their cost)
    oilProd_revenue = (fopt_Nc[1:n] - fopt_Nc[0:n-1]) * co 
    waterInj_cost = (fwit_Nc[1:n] - fwit_Nc[0:n-1]) * cwi
    waterProd_cost = (fwpt_Nc[1:n] - fwpt_Nc[0:n-1]) * cw
    #discounted time
    t = [1/((1 + b)**(Tn[i+1]/365)) for i in range(n-1)]
    t = np.array(t)
    #profit
    profit = oilProd_revenue - waterInj_cost - waterProd_cost
    #calculate NPV
    npv = np.dot(t,profit)

    return npv
    

def CalculateNPV_with_Simulation_Timestep_v2(file_names, economic_parameters):
    WOPR, WWPR, WWIR, time_vec = GetWellRates(file_names)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]

    oilP_revenue = np.sum(co*WOPR, axis = 1, keepdims=True) #sum oil revenue accross all producers
    waterP_cost = np.sum(cw*WWPR, axis = 1, keepdims=True) #sum the water treatment costs across all producers 
    waterI_cost = np.sum(cwi*WWIR, axis = 1, keepdims=True) #sum injection cost across all injectors
    profit = oilP_revenue - waterP_cost - waterI_cost
    # #get time step size from the second entry in Tn(total time at the end of the nth time step)
    Tn = np.array(time_vec)
    time_steps = [Tn[i] - Tn[i-1] for i in range(1, len(time_vec))]
    t = (1 + b)**(Tn/365)
    const = time_steps / t[1:]
    npv = np.dot(const, profit) 
    
    return npv    

def CalculateNPV_with_UserDefined_Timestep_v2(file_names, economic_parameters, timestep_size, time_elapsed):
    #TODO not implemented
    wopr, wwpr, wwir, time_vec = GetWellRates(file_names)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]
    dt = timestep_size #control time step
    
    Total_Days = time_elapsed
    Nc = int(Total_Days/dt) #Number of control steps
    time_steps = dt*np.ones(Nc) #time steps in an array
    Tn = np.cumsum(time_steps, axis=0)
    #print(Tn_20)
    ## indices of the required T_n in the original time_vec.  
    ##=========I had to implement this way because the simulation timestep could change due to convergence issues, thereby resulting in more timesteps than necessary===##
    # Initialize an empty set to store unique values encountered
    seen_values = set()
    # Initialize an empty list to store indices of required T_n
    indices = []
    # Iterate over time_vec
    for i, value in enumerate(time_vec):
        # Check if the value is in Tn and not already seen
        if value in Tn and value not in seen_values:
            # Add the index to indices
            indices.append(i)
            # Add the value to seen_values
            seen_values.add(value)
    #print(indices)
    assert len(indices)==len(Tn), "size of total time and indices should be equal"
    num_of_entries_at_timestep = 1 + np.array(indices) #number of entries to average over. Turns out this is  = index + 1
    num_of_entries_at_timestep = num_of_entries_at_timestep.reshape(-1,1) #ensures size is in (Nc, 1) for correct division

    #wopr
    # Calculate the cumulative sum of each column up to the ith row
    WOPR_cumulative_sum = np.cumsum(wopr, axis=0)
    WOPR_cumulative_sum_Nc = np.array([WOPR_cumulative_sum[i] for i in indices])
    WOPR_ave_Nc = WOPR_cumulative_sum_Nc/num_of_entries_at_timestep

    #WWPR
    WOPR_cumulative_sum = np.cumsum(wwpr, axis=0)
    WWPR_cumulative_sum_Nc = np.array([WOPR_cumulative_sum[i] for i in indices])
    WWPR_ave_Nc = WWPR_cumulative_sum_Nc/num_of_entries_at_timestep

    #WWPR
    WWIR_cumulative_sum = np.cumsum(wwir, axis=0)
    WWIR_cumulative_sum_Nc = np.array([WWIR_cumulative_sum[i] for i in indices])
    WWIR_ave_Nc = WWIR_cumulative_sum_Nc/num_of_entries_at_timestep

    #calculate NPV
    oilP_revenue= np.sum(co*WOPR_ave_Nc, axis = 1, keepdims=True) #sum oil revenue accross all producers
    waterP_cost= np.sum(cw*WWPR_ave_Nc, axis = 1, keepdims=True) #sum the water treatment costs across all producers 
    waterI_cost= np.sum(cwi*WWIR_ave_Nc, axis = 1, keepdims=True) #sum injection cost across all injectors
    profit = oilP_revenue - waterP_cost - waterI_cost
    t = (1 + b)**(Tn/365)
    const= time_steps / t
    npv = np.dot(const, profit) 
    
    return NotImplemented

def CalculateNPV_with_Simulation_Timestep(file_names, economic_parameters):
    WOPR, WWPR, WWIR, time_vec = GetWellRates(file_names)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]
    # Calculate the cumulative sum of each column up to the ith row
    WOPR_cumulative_sum = np.cumsum(WOPR, axis=0)
    # Calculate the average of each column from the first row to the ith row
    WOPR_ave = WOPR_cumulative_sum[1:] / np.arange(2, WOPR.shape[0] + 1)[:, None]

    # Calculate the cumulative sum of each column up to the ith row
    WWPR_cumulative_sum = np.cumsum(WWPR, axis=0)
    # Calculate the average of each column from the first row to the ith row
    WWPR_ave = WWPR_cumulative_sum[1:] / np.arange(2, WWPR.shape[0] + 1)[:, None]

    # Calculate the cumulative sum of each column up to the ith row
    WWIR_cumulative_sum = np.cumsum(WWIR, axis=0)
    # Calculate the average of each column from the first row to the ith row
    WWIR_ave = WWIR_cumulative_sum[1:] / np.arange(2, WWIR.shape[0] + 1)[:, None]

    oilP_revenue = np.sum(co*WOPR_ave, axis = 1, keepdims=True) #sum oil revenue accross all producers
    waterP_cost = np.sum(cw*WWPR_ave, axis = 1, keepdims=True) #sum the water treatment costs across all producers 
    waterI_cost = np.sum(cwi*WWIR_ave, axis = 1, keepdims=True) #sum injection cost across all injectors
    profit = oilP_revenue - waterP_cost - waterI_cost
    # #get time step size from the second entry in Tn(total time at the end of the nth time step)
    Tn = np.array(time_vec)
    time_steps = [Tn[i] - Tn[i-1] for i in range(1, len(time_vec))]
    t = (1 + b)**(Tn/365)
    const = time_steps / t[1:]
    npv = np.dot(const, profit) 
    
    return npv

def CalculateNPV_with_UserDefined_Timestep(file_names, economic_parameters, timestep_size, time_elapsed):
    wopr, wwpr, wwir, time_vec = GetWellRates(file_names)
    co  = economic_parameters["co"]
    cw  = economic_parameters["cw"]
    cwi = economic_parameters["cwi"]
    b   = economic_parameters["b"]
    dt = timestep_size #control time step
    
    Total_Days = time_elapsed
    Nc = int(Total_Days/dt) #Number of control steps
    time_steps = dt*np.ones(Nc) #time steps in an array
    Tn = np.cumsum(time_steps, axis=0)
    #print(Tn_20)
    ## indices of the required T_n in the original time_vec. I needed this to know position of the average rate at the every timestep    
    ##=========I had to implement this way because the simulation timestep could change due to convergence issues, thereby resulting in more timesteps than necessary===##
    # Initialize an empty set to store unique values encountered
    seen_values = set()
    # Initialize an empty list to store indices of required T_n
    indices = []
    # Iterate over time_vec
    for i, value in enumerate(time_vec):
        # Check if the value is in Tn and not already seen
        if value in Tn and value not in seen_values:
            # Add the index to indices
            indices.append(i)
            # Add the value to seen_values
            seen_values.add(value)
    #print(indices)
    assert len(indices)==len(Tn), "size of total time and indices should be equal"
    num_of_entries_at_timestep = 1 + np.array(indices) #number of entries to average over. Turns out this is  = index + 1
    num_of_entries_at_timestep = num_of_entries_at_timestep.reshape(-1,1) #ensures size is in (Nc, 1) for correct division

    #wopr
    # Calculate the cumulative sum of each column up to the ith row
    WOPR_cumulative_sum = np.cumsum(wopr, axis=0)
    WOPR_cumulative_sum_Nc = np.array([WOPR_cumulative_sum[i] for i in indices])
    WOPR_ave_Nc = WOPR_cumulative_sum_Nc/num_of_entries_at_timestep

    #WWPR
    WOPR_cumulative_sum = np.cumsum(wwpr, axis=0)
    WWPR_cumulative_sum_Nc = np.array([WOPR_cumulative_sum[i] for i in indices])
    WWPR_ave_Nc = WWPR_cumulative_sum_Nc/num_of_entries_at_timestep

    #WWPR
    WWIR_cumulative_sum = np.cumsum(wwir, axis=0)
    WWIR_cumulative_sum_Nc = np.array([WWIR_cumulative_sum[i] for i in indices])
    WWIR_ave_Nc = WWIR_cumulative_sum_Nc/num_of_entries_at_timestep

    #calculate NPV
    oilP_revenue= np.sum(co*WOPR_ave_Nc, axis = 1, keepdims=True) #sum oil revenue accross all producers
    waterP_cost= np.sum(cw*WWPR_ave_Nc, axis = 1, keepdims=True) #sum the water treatment costs across all producers 
    waterI_cost= np.sum(cwi*WWIR_ave_Nc, axis = 1, keepdims=True) #sum injection cost across all injectors
    profit = oilP_revenue - waterP_cost - waterI_cost
    t = (1 + b)**(Tn/365)
    const= time_steps / t
    npv = np.dot(const, profit) 
    
    return npv

def GetBHPs(file_names):
    bhp_filename = file_names["WBHP"]
    BHPs, _ = Extract_BHP_DataMatrix(bhp_filename)
    return np.array(BHPs)


def GetWellRates(file_names):
    matrices = []
    time_vec = None
    items = list(file_names.items())
    for i in range(len(items) - 1):  #loop excludes the last entries of file_names because it is WBHP and welltype = ' H' is useless in extract_wellrates_datamatrix
        keys, file_name = items[i]
        well_type = ' '+ keys[2]
        matrix, time_vec = Extract_WellRates_DataMatrix(file_name, well_type)
        matrices.append(matrix)

    #well oil production rate
    wopr = np.array(matrices[0])
    #well water production rate
    wwpr = np.array(matrices[1])
    #well water injection rate
    wwir = np.array(matrices[2])
    #simulation timestep
    Tn = np.array(time_vec)
   
    return wopr, wwpr, wwir, Tn

def GetFieldTotalProduction(file_name):
    matrix = []
    time_vec = None
    matrix, time_vec = Extract_FieldTotalProduction_DataMatrix(file_name)
    #cummulative field water injection total
    matrix = np.array(matrix)
    fwit = matrix[:,0]
    #cummulative field oil production total
    fopt = matrix[:,1]
    #cummulative field water production total
    fwpt = matrix[:,2]
    #cummulative field liquid production total
    flpt = matrix[:,3]
    #cummulative field liquid production total
    fgpt = matrix[:,4]
    #simulation timestep
    Tn = np.array(time_vec)
   
    return fopt, fwit, fwpt, flpt, fgpt, Tn

def Extract_WellRates_DataMatrix(file_name, well_type = ' P' ):
    # Initialize empty arrays to store the data
    arrays = []

    # Open the CSV file and read its contents
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # append each row to the arrays list
            array = [num for num in row]
            arrays.append(array)
    #The second column contains the simulation timestep. I extract the time from this second column, starting from the second row down to the last row
    time_vector = [float(row[1]) for row in arrays[1:]]
    # Find the indices of columns where the entry on the first row starts with 'P' or 'I', depending on the well_type
    indices = [i for i, entry in enumerate(arrays[0]) if entry.startswith(well_type)] #there's a space before P and I

    # Extract vectors from the respective columns where 'P' starts
    matrix = []
    for row in arrays[1:]:
        vector = [float(row[i]) for i in indices]
        matrix.append(vector)

    assert len(time_vector) == len(matrix), "unequal rows: time and data should have equal number of entries"
    
    return matrix, time_vector

def Extract_FieldTotalProduction_DataMatrix(file_name):
    # Initialize empty arrays to store the data
    arrays = []

    # Open the CSV file and read its contents
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # append each row to the arrays list
            array = [num for num in row]
            arrays.append(array)
    #The second column contains the simulation timestep. I extract the time from this second column, starting from the second row down to the last row
    time_vector = [float(row[1]) for row in arrays[1:]]

    matrix = []
    for row in arrays[1:]:
        vector = [float(row[i]) for i in range(2,7)]    #hardcorded this because FWIT starts at column index 2 and last entry (FGPT) ends at column index 6
        matrix.append(vector)

    assert len(time_vector) == len(matrix), "unequal rows: time and data should have equal number of entries"
    
    return matrix, time_vector

def Extract_BHP_DataMatrix(file_name):
    # Initialize empty arrays to store the data
    arrays = []

    # Open the CSV file and read its contents
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # append each row to the arrays list
            array = [num for num in row]
            arrays.append(array)
    #The second column contains the simulation timestep. I extract the time from this second column, starting from the second row down to the last row
    time_vector = [float(row[1]) for row in arrays[1:] ]

    matrix = []
    for row in arrays[1:]:
        vector = [float(entry) for entry in row[2:]]
        matrix.append(vector)

    assert len(time_vector) == len(matrix), "unequal rows: time and data should have equal number of entries"
    
    return matrix, time_vector

def delete_files() -> None:
    """
    Delete files with specific extensions in a directory.

    Parameters:
    - directory (str): Path to the directory containing files to delete.
    - extensions (list): List of extensions of the files to delete (e.g., ['txt', 'csv']).

    Returns:
    - None
    """
    extensions = [".csv", ".gsg", ".afi",
                    ".ixf", ".dbprtx", ".default.session",
                    ".default.sessionlock", ".EGRID",".FINIT"
                        ,".FINSPEC", ".FRSSPEC", ".FSMSPEC",
                        ".FUNRST", ".FUNSMRY",".h5",
                        ".MSG", ".PRT", ".PRTX",".REP",
                        ".RTELOG", ".RTEMSG",".ixf",".bat" , ".afo_xml", ".afo"]
    try:
        # List all files in the directory
        files = os.listdir(os.getcwd())
        
        # Filter files with the specified extensions
        files_to_delete = [file for file in files if any(file.endswith(f'{ext}') for ext in extensions)]

        # Delete each file with the specified extensions
        for file in files_to_delete:
            file_path = os.path.join(os.getcwd(), file)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting files: {e}")

    return None

def delete_files_2() -> None:
    """
    Delete files with specific extensions in a directory, and folders that start with "restart".

    Returns:
    - None
    """
    import shutil

    extensions = [".csv", ".gsg", ".afi",
                    ".ixf", ".dbprtx", ".default.session",
                    ".default.sessionlock", ".EGRID",".FINIT"
                        ,".FINSPEC", ".FRSSPEC", ".FSMSPEC",
                        ".FUNRST", ".FUNSMRY",".h5",
                        ".MSG", ".PRT", ".PRTX",".REP",
                        ".RTELOG", ".RTEMSG",".ixf",".bat" , 
                        ".afo_xml", ".afo", ".lock", ".lock-journal"]
    try:
        # List all files and directories in the current directory
        files_and_dirs = os.listdir(os.getcwd())

        # Filter files with the specified extensions
        files_to_delete = [entry for entry in files_and_dirs if any(entry.endswith(f'{ext}') for ext in extensions)]

        # Delete each file with the specified extensions
        for file in files_to_delete:
            file_path = os.path.join(os.getcwd(), file)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

        # List directories starting with "restart"
        dirs_to_delete = [entry for entry in files_and_dirs if os.path.isdir(entry) and entry.startswith("restart")]

        # Delete each directory starting with "restart"
        for directory in dirs_to_delete:
            dir_path = os.path.join(os.getcwd(), directory)
            shutil.rmtree(dir_path)
            print(f"Deleted directory: {dir_path}")

    except Exception as e:
        print(f"Error deleting files: {e}")

    return None