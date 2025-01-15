import copy
import sys, os
sys.path.append(os.getcwd())
from sympy import true
from DRL.Utilities import *
import uuid
import numpy as np
import os
import shutil

##This INTERSECT Automated simulator is written for Deep Reinforcement Learning application
class ReservoirSimulator_HFS:
     #dictionary to keep track of instances created from this class using the same "sim_inputs" 
    instances = {}
    instances_count = 0
    def __init__(self, sim_inputs):
        ReservoirSimulator_HFS.instances_count += 1

        #generate unique id for instance of this class
        self.UNIQUE_ID = str(uuid.uuid4())
        self.instance_id = self.get_instance_id(sim_inputs)
        if self.instance_id not in self.instances:
            self.instances[self.instance_id] = 1
        else:
            self.instances[self.instance_id] += 1
        self.instance_count = self.get_instance_count()
        self.initialize_variables(sim_inputs)




    def get_instance_id(self, input_data):
        hashable = tuple(input_data)
        return hash(hashable)  # Generate a unique identifier based on input data

    def get_instance_count(self):
        return self.instances.get(self.instance_id, 0)
  

    def initialize_variables(self, sim_inputs) -> None:

        res_param = copy.deepcopy(sim_inputs)

        #newly added
        self.realzID = sim_inputs["realzID"]
        self.num_realization = sim_inputs["num_realizations"]

        #set well and control bounds
        self.num_prod = res_param["num_prod"] 
        self.num_inj = res_param["num_inj"]  
        self.num_wells = self.num_inj + self.num_prod
        self.water_inj_bounds = res_param["water_inj_bounds"] 
        self.prod_bhp_bounds = res_param["prod_bhp_bounds"] 
        self.wells_lower_bound = res_param["wells_lower_bound"] 
        self.wells_upper_bound = res_param["wells_upper_bound"] 
    
        #set timing
        self.time_increment = res_param["time_increment"]  
        self.control_step_size = res_param["control_step_size"] 
        self.num_stepping_iter = res_param["num_stepping_iter"] 
        self.num_control_steps = res_param["num_control_steps"] 
        self.total_days = res_param["total_days"]
        self.time_elapsed = 0.0
        self.old_time_elapsed = 0.0
        self.control_step_id = 0
        self.num_run_per_control_step = 11
        self.restart = res_param["restart"]
 

        #economic parameters
        self.economic_param = {"co":  res_param["oil_price"],
                                "cw": res_param["water_prod_cost"],
                                "cwi":res_param["water_inj_cost"],
                                 "b": res_param["discount_factor"]} 
        self.NPVs = []  #cummulative NPV until the recent control step

        
        #computational resources
        self.np = res_param["num_processors"] #number of processor
        
        #input/output files
        self.directory_copy_files = res_param["directory"]
        self.schedule_file = "schedule_file_" + self.UNIQUE_ID + ".ixf"
        self.AFI_file = "BASE_" + self.UNIQUE_ID + ".afi"
        self.ECL2IX_file = "BASE_ECL2IX_" + self.UNIQUE_ID +".ixf"
        self.gsg_file = "BASE_" + str(self.realzID) + "_" + self.UNIQUE_ID +".gsg"
        self.res_edit_file = "BASE_res_edits_" + self.UNIQUE_ID + ".ixf"
        self.fm_edit_file = "BASE_fm_edits_" + self.UNIQUE_ID + ".ixf"
        self.report_setting_file = "BASE_report_" + self.UNIQUE_ID + ".ixf"
        self.group_name = res_param["group_name"]
        self.batch_file = "batch_ix_"+ self.UNIQUE_ID + ".bat"
        self.npv_files = {  "WOPR": "BASE_WOPR_"+ self.UNIQUE_ID + ".csv",
                            "WWPR": "BASE_WWPR_"+ self.UNIQUE_ID + ".csv",
                            "WWIR": "BASE_WWIR_"+ self.UNIQUE_ID + ".csv",
                            "WBHP": "BASE_WBHP_"+ self.UNIQUE_ID + ".csv",
                            "NPV" : "BASE_NPVDATA_" + self.UNIQUE_ID + ".csv",
                            "Ave_WBLOCK_PRESS": "BASE_WBLOCK_PRESS_" + self.UNIQUE_ID + ".csv"}
        self.custom_file_name  = "CustomScripts_" + self.UNIQUE_ID + ".ixf"
        self.NPV_file_name = "CustomNPVScript_" + self.UNIQUE_ID + ".ixf"
        self.generate_custom_script() #generate ixf file for observed well data used in RL 
        self.generate_NPV_script() #This has replaced the custom_script file for NPV calculation
        self.generate_ecl2ix_file()
        self.generate_gsg_file()
        self.generate_res_edit_file()
        self.generate_report_file()
        self.generate_AFI_file() #generate the main afi file for reservoir simulation
        self.generate_batch_file() #generate batch file for automatic simulation

        #start generating schedule.ixf file
        schedule_SOF(self.num_inj, self.num_prod, self.group_name, self.schedule_file)
        fm_edits_SOF(self.fm_edit_file) #used for restarting simulation
        self.afi_restart_string = "" #empty string to help initialize variable in append_restart instruction to afi file
        self.isStart = True  #bool to decide whether or not afi file will be appended with a restart instruction. if True, no instruction is written

        return None

    def generate_res_edit_file(self) -> None:
        #generate reservoir_edits.ixf 
        with open(self.res_edit_file, "w") as _file:
            init_timestep = 0.5 #18
            min_timestep = 0.1
            max_timestep = 15 #30
            _file.write('\nMODEL_DEFINITION\n\n'
                        f'TimeStepSizingControls {{\n'
                        f'\tInitialTimeStep={init_timestep}\n'
                        f'\tMinTimeStep={min_timestep}\n'
                        f'\tMaxTimeStep={max_timestep}\n'
                        f'}}')
        
        return None
            
    def generate_ecl2ix_file(self) -> None:

        src_file = self.directory_copy_files + "\BASE_ECL2IX_IX2.ixf"
        try:
            shutil.copyfile(src_file, self.ECL2IX_file)
        except Exception as e:
            print(f"Error copying file: {e}")
        
        return None

    def generate_gsg_file(self) -> None:

        src_file = self.directory_copy_files + "\BASE_" + str(self.realzID) + ".gsg"
        try:
            shutil.copyfile(src_file, self.gsg_file)
        except Exception as e:
            print(f"Error copying file: {e}")
        
        return None

    def generate_report_file(self) -> None:  

        src_file = self.directory_copy_files + "\BASE_report_settings.ixf"
        try:
            shutil.copyfile(src_file, self.report_setting_file)
        except Exception as e:
            print(f"Error copying file: {e}")
        
        return None

    def generate_AFI_file(self) -> None:

        simulation_name = self.AFI_file
        # Define the content template with placeholders
        content_template ='#################################\n'
        content_template += '# IXFVERSION: 2020.4 (20201208.1)\n'
        content_template += '#################################\n'
        content_template += f'SIMULATION ix "{simulation_name}" {{\n'
        content_template += f'  INCLUDE "{self.gsg_file}" {{ type="gsg" gsg_type="geom_and_props" }}\n'
        content_template += f'  INCLUDE "{self.ECL2IX_file}" {{ type="ixf" }}\n'
        content_template += f'  INCLUDE "{self.res_edit_file}" {{ type="ixf" preserve="True" }}\n'
        content_template += f'  INCLUDE "{self.fm_edit_file}" {{ type="ixf" preserve="True" }}\n'
        content_template +=  '  EXTENSION "EGRIDReport"\n}\n'
        content_template += 'SIMULATION fm {\n'
        content_template += f'  INCLUDE "{self.custom_file_name}" {{type="ixf"}}\n'
        content_template += f'  INCLUDE "{self.NPV_file_name}" {{type="ixf"}}\n'
        content_template += f'  INCLUDE "{self.report_setting_file}" {{type="ixf"}}\n'
        content_template += f'  INCLUDE "{self.schedule_file}"\n'    
        content_template +=  '  EXTENSION "custom_scripts"\n}\n'
        
        with open(simulation_name, 'w') as file:
            file.write(content_template)
        
        return None

    def generate_NPV_script(self) -> None: 
        # Write the content to the file
        with open(self.NPV_file_name, 'w') as file:
            file.write("MODEL_DEFINITION \n\n\n\n")
            file.write(self.generate_NPV_content())
        
        return None

    def generate_custom_script(self) -> None:
        custom_file_name  = self.custom_file_name
  
        # Write the content to the file
        with open(custom_file_name, 'w') as file:
            # file.write(self.generate_custom_control_content("WLPT", "LIQUID_PRODUCTION_CUML"))
            # file.write("\n\n")
            # file.write(self.generate_custom_control_content("WWPT", "WATER_PRODUCTION_CUML"))
            # file.write("\n\n")
            file.write("MODEL_DEFINITION \n\n\n\n")
            file.write(self.generate_custom_control_content("WOPR", "OIL_PRODUCTION_RATE"))
            file.write("\n\n")
            file.write(self.generate_custom_control_content("WWIR", "WATER_INJECTION_RATE"))  
            file.write("\n\n")
            file.write(self.generate_custom_control_content("WWPR", "WATER_PRODUCTION_RATE"))    
            file.write("\n\n")
            file.write(self.generate_custom_control_content("WBHP", "BOTTOM_HOLE_PRESSURE")) 
            file.write("\n\n")
            file.write(self.generate_custom_control_content("Ave_WBLOCK_PRESS", "WELL_BLOCK_AVERAGE_PRESSURE")) 

        return None 

    
    def generate_custom_control_content(self, key, prop_variable) -> str:
        # Define the variables
        control_name = "Export" + key + "Data"
        execution_position = "BEGIN_TIMESTEP"
        filename_variable = self.npv_files[key]
        property_variable = prop_variable #'BOTTOM_HOLE_PRESSURE'
        field_name = 'FM_FIELD'
        if self.restart and self.control_step_id > 0:
            action = "a"
            initial_string = ''
        else:
            action = "w"
            initial_string = "print('DATE, TIME', end='', file=_file)\n"
            initial_string += 'for well in _current_wells:\n'
            initial_string += "    print(',', well.get_name(), end='', file=_file)\n"
            initial_string += "print('\\n', end='', file=_file)"

        # Construct the control content with placeholders
        control_content = f'CustomControl "{control_name}" {{\n\n' 
        control_content += f'    ExecutionPosition={execution_position}\n\n'
        control_content +=  '    InitialScript=@{\n'
        control_content += f"filename = '{filename_variable}'\n"
        control_content += f'_file = open(filename, "{action}")\n'
        control_content += "_current_wells = []\n"
        control_content += f"for well in Group('{field_name}').FlowEntities.get():\n"
        control_content +=  "   if well not in _current_wells:\n"
        control_content +=  "       _current_wells.append(well)\n"
        control_content += f"{initial_string}\n"
        control_content += "}@\n\n"
        control_content += "    Script=@{\n"
        control_content += "print(FieldManagement().CurrentDate.get(), end='', file=_file)\n"
        control_content += "print(',', FieldManagement().CurrentTime.get(), end='', file=_file)\n\n"
        control_content += "for well in _current_wells:\n"
        control_content += f"    print(',', Well(well.get_name()).get_property({property_variable}).value, end='', file=_file)\n"
        control_content += "print('\\n', end='', file=_file)\n"
        control_content += "}@\n\n"
        control_content += "    FinalScript=@{\n"
        control_content += "print(FieldManagement().CurrentDate.get(), end='', file=_file)\n"
        control_content += "print(',', FieldManagement().CurrentTime.get(), end='', file=_file)\n"
        control_content += "for well in _current_wells:\n"
        control_content += f"    print(',', Well(well.get_name()).get_property({property_variable}).value, end='', file=_file)\n"
        control_content += "_file.close()\n"
        control_content += f'print("File", "{filename_variable}",'
        control_content += f' "is written for {key} data.")\n'
        control_content += '}@\n\n'
        control_content += '}'
    
        return control_content


    def generate_NPV_content(self) -> None:
        # Define the variables
        control_name = "ExportDataForNPV"
        execution_position = "BEGIN_TIMESTEP"
        filename_variable = self.npv_files["NPV"]
        field_name = 'FM_FIELD'

        if self.restart and self.control_step_id > 0:
            action = "a"
            initial_string = ''
        else:
            action = "w"
            initial_string = "print('DATE, TIME', end='', file=_file)\n"
            initial_string += 'for parameter in _parameters:\n'
            initial_string += "	    print(',', parameter, end='', file=_file)\n"
            initial_string += "print('\\n', end='', file=_file)"

        # Construct the control content with placeholders
        NPV_script_content = f'CustomControl "{control_name}" {{\n\n'
        NPV_script_content += f'ExecutionPosition={execution_position}\n'
        NPV_script_content += '    InitialScript=@{\n'
        NPV_script_content += f"filename = '{filename_variable}'\n"
        NPV_script_content += f"_file = open(filename, '{action}')\n"
        NPV_script_content += "_parameters = ['FWIT', 'FOPT', 'FWPT', 'FLPT', 'FGPT']\n"
        NPV_script_content += "_parameter_keys = ['WATER_INJECTION_CUML', 'OIL_PRODUCTION_CUML', 'WATER_PRODUCTION_CUML', 'LIQUID_PRODUCTION_CUML', 'GAS_PRODUCTION_CUML']\n"
        NPV_script_content += f"{initial_string}\n"
        NPV_script_content += "}@\n\n"
        NPV_script_content += "    Script=@{\n"
        NPV_script_content += "print(FieldManagement().CurrentDate.get(), end='', file=_file)\n"
        NPV_script_content += "print(',', FieldManagement().CurrentTime.get(), end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(WATER_INJECTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(OIL_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(WATER_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(LIQUID_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(GAS_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += "print('\\n', end='', file=_file)\n"
        NPV_script_content += "}@\n\n"
        NPV_script_content += "    FinalScript=@{\n"
        NPV_script_content += "print(FieldManagement().CurrentDate.get(), end='', file=_file)\n"
        NPV_script_content += "print(',', FieldManagement().CurrentTime.get(), end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(WATER_INJECTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(OIL_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(WATER_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(LIQUID_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += f"print(',', Group('{field_name}').get_property(GAS_PRODUCTION_CUML).value, end='', file=_file)\n"
        NPV_script_content += "_file.close()\n"
        NPV_script_content += f'print("File", "{filename_variable}",'
        NPV_script_content += f' "is written for NPV data.")\n'
        NPV_script_content += '}@\n\n'
        NPV_script_content += '}'

        return NPV_script_content


    def remove_last_line_npv_files(self) -> None:

        filename_variable = self.npv_files
        for _, filename_variable in self.npv_files.items():
            with open(filename_variable, 'r') as file:
                lines  = file.readlines()
            if lines:
                lines.pop()
            with open(filename_variable, 'w') as file:
                file.writelines(lines)

        return None

    def generate_batch_file(self) -> None:
        # Define the variables
        np_value = f'{self.np}'
        afi_file = f'{self.AFI_file}'

        # Construct the batch content with placeholders
        batch_content = '@echo off\n'
        batch_content += 'SET LSHORST=no-net\n'
        batch_content += '@echo off\n\n'
        batch_content += f'"C:\\ecl\\macros\\eclrun.exe" -dd -v 2024.3 --np {np_value} ix {afi_file}\n\n'
        batch_content += 'CLS\n'
        batch_content += 'EXIT'

        # Specify the file path
        file_path = self.batch_file

        # Write the content to the file
        with open(file_path, 'w') as file:
            file.write(batch_content)

        return None


    
    def reset_variables(self) -> None:

        self.control_step_id = 0
        self.time_elapsed = 0.0
        self.old_time_elapsed = 0.0
        self.num_run_per_control_step = 11
        self.NPVs = []
        self.afi_restart_string = ""
        self.isStart = True
        erase(self.schedule_file)
        schedule_SOF(self.num_inj, self.num_prod, self.group_name, self.schedule_file)
        erase(self.fm_edit_file)
        self.generate_AFI_file()
        self.generate_custom_script()
        self.generate_NPV_script()
        fm_edits_SOF(self.fm_edit_file)

        return None

    #this method is specific to running one step in the Deep Reinforcement Learning reservoir environment
    def run_one_control_step(self, well_controls) -> None:
        try:
            if self.restart:
                ## create the schedule.ixf file             
                append_ControlStep(self.control_step_id + 1, well_controls, self.num_inj, self.num_prod, self.schedule_file)
                self.old_time_elapsed = self.time_elapsed
                self.time_elapsed = append_Time(self.num_stepping_iter, self.time_increment, self.time_elapsed, self.schedule_file)
                ##append the restart time in the fm_edit file for the next control step
                restart_string = append_restart(self.time_elapsed, self.UNIQUE_ID, self.fm_edit_file)
                ##append end of file for schedule.ixf. This completes the generation of schedule.ixf
                append_EOF(self.schedule_file)
                
                if self.control_step_id > 0 and self.control_step_id < self.num_control_steps:
                    self.remove_last_line_npv_files()
                ## run simulation from the latest time till the next control step
                run_simulation(self.batch_file, self.control_step_id + 1)
                ##erase any restart instruction in the afi file. if isStart =True, no action is performed
                erase_EOF_afi(self.afi_restart_string, self.AFI_file, self.isStart)
                self.isStart = False

                ##update control step ids and regenerate custom script that appends latest data to existing ones
                self.control_step_id += 1
                self.generate_custom_script()
                self.generate_NPV_script()

                ##append restart instruction in the afi file. 
                self.afi_restart_string = append_restart_afi(self.time_elapsed, self.UNIQUE_ID, self.AFI_file)

                ## Since only one restart is required after a control step, erase previous restart time so that new restart time can be written
                erase_EOF_fm_edit(restart_string, self.fm_edit_file)
                ##erase the End of file keyword in schedule.ixf so that new well controls can be written
                erase_EOF(self.schedule_file)  
            else:
                # create the schedule_python.ixf file
                append_ControlStep(self.control_step_id + 1, well_controls, self.num_inj, self.num_prod, self.schedule_file)
                self.old_time_elapsed = self.time_elapsed
                self.time_elapsed = append_Time(self.num_stepping_iter, self.time_increment, self.time_elapsed, self.schedule_file)
                append_EOF(self.schedule_file)

                ## run simulation with the latest schedule - note this runs from the very start of the simulation till the latest time for the new well_controls. 
                ## Need to use a RESTART to avoid repetitive computation!!
                run_simulation(self.batch_file, self.control_step_id + 1)

                ##update control step ids
                self.control_step_id += 1

                ##erase the End of file keyword in schedule_python.ixf so that new well controls can be written
                erase_EOF(self.schedule_file)  
        except Exception:
            print("Simulation Failed!!")
       
        return None

    

    def get_npv(self):
        ##calculate npv from the beginning to the latest time
        npv_till_latest_time = self.CalculateNPV() 
        self.NPVs.append(npv_till_latest_time)  
        if self.control_step_id - 1 == 0:
            npv = npv_till_latest_time
        else:
            npv = npv_till_latest_time - self.NPVs[self.control_step_id - 2]

        # if self.restart:
        #     #remove last lines from npv files so that new production data can be appended.
        #     # last time  of npv file is usually the start time for next control step so I do this to prevent duplicacy
        #     self.remove_last_line_npv_files()

        return npv
    
    
    def CalculateNPV(self):
        #return CalculateNPV_with_UserDefined_Timestep(self.npv_files, self.economic_param, self.control_step_size, self.time_elapsed)
        return CalculateNPV_FieldCumProd(self.npv_files["NPV"], self.economic_param, use_simulation_timestep = False, timestep_size = self.control_step_size, total_days = self.time_elapsed)



    def get_observation(self):
        wopr, wwpr, wwir, time_vec = GetWellRates(self.npv_files)
        BHPs = GetBHPs(self.npv_files)

        ##Handles cases when the simulation cuts timestep to enable convergence..In this cases, time_vec is usually more than required
        if len(time_vec) is not self.num_run_per_control_step:
            from scipy.interpolate import CubicSpline
            
            ## Timevec required to match the number of run per control step 
            time_vec_new = np.linspace(self.old_time_elapsed, self.time_elapsed, self.num_run_per_control_step)
            
            ##Discard repititions in time_vec and take the mean of values in welldata for which time_vec is repeated . This is done to ensure cubic spline works fine
            unique_time_vec, idx = np.unique(time_vec, return_inverse=True)
            unique_wopr = np.array([np.mean(wopr[idx == i, :], axis=0) for i in range(len(unique_time_vec))])
            unique_wwpr = np.array([np.mean(wwpr[idx == i, :], axis=0) for i in range(len(unique_time_vec))])
            unique_wwir = np.array([np.mean(wwir[idx == i, :], axis=0) for i in range(len(unique_time_vec))])
            unique_bhp = np.array([np.mean(BHPs[idx == i, :], axis=0) for i in range(len(unique_time_vec))])

            #form cubic spline
            cs_wopr = CubicSpline(unique_time_vec, unique_wopr)
            cs_wwpr = CubicSpline(unique_time_vec, unique_wwpr)
            cs_wwir = CubicSpline(unique_time_vec, unique_wwir)
            cs_bhp = CubicSpline(unique_time_vec, unique_bhp)

            #interpolate for the required time_vec
            time_vec = time_vec_new
            wopr = cs_wopr(time_vec_new)
            wwpr = cs_wwpr(time_vec_new)
            wwir = cs_wwir(time_vec_new)
            BHPs = cs_bhp(time_vec_new)
                     
        indices = np.where(time_vec == self.old_time_elapsed)[0]
        if len(indices) > 0:
            index_start = indices[0]
        else:
            # Handle the case where self.old_time_elapsed is not found in time_vec
            print("self.old_time_elapsed not found in time_vec")
            print("time_vec: ", time_vec)
            print("old elapsed time: ", self.old_time_elapsed)
            print("Reservoir with issue: ", self.UNIQUE_ID)
        index_end = np.where(time_vec == self.time_elapsed)[0][0]
        wopr_current = wopr[index_start:index_end+1,:]
        wwpr_current = wwpr[index_start:index_end+1,:]
        wwir_current = wwir[index_start:index_end+1,:]
        bhps_current = BHPs[index_start:index_end+1,:]
        assert wopr_current.shape[0] == wwpr_current.shape[0] == wwir_current.shape[0] == bhps_current.shape[0], "all observations must have the same length"
        self.num_run_per_control_step = wopr_current.shape[0]
        obs_data = np.array([]).reshape(0, self.num_run_per_control_step)
        obs_data = np.vstack((obs_data, wopr_current.T))
        obs_data = np.vstack((obs_data, wwpr_current.T))
        obs_data = np.vstack((obs_data, wwir_current.T))
        obs_data = np.vstack((obs_data, bhps_current.T))
        unscaled = obs_data.T.flatten()
        scaled = (unscaled - min(unscaled)) / (max(unscaled) - min(unscaled))  #min-max normalization

        return scaled
    

    
    def delete_files(self) -> None:
        """
        Delete files with specific extensions in a directory.

        Parameters:
        - directory (str): Path to the directory containing files to delete.
        - extensions (list): List of extensions of the files to delete (e.g., ['txt', 'csv']).

        Returns:
        - None
        """
        extensions = [f"{self.UNIQUE_ID}.csv", f"{self.UNIQUE_ID}.gsg", f"{self.UNIQUE_ID}.afi",
                       f"{self.UNIQUE_ID}.ixf", f"{self.UNIQUE_ID}.dbprtx", f"{self.UNIQUE_ID}.default.session",
                        f"{self.UNIQUE_ID}.default.sessionlock", f"{self.UNIQUE_ID}.EGRID",f"{self.UNIQUE_ID}.FINIT"
                         ,f"{self.UNIQUE_ID}.FINSPEC", f"{self.UNIQUE_ID}.FRSSPEC", f"{self.UNIQUE_ID}.FSMSPEC",
                          f"{self.UNIQUE_ID}.FUNRST", f"{self.UNIQUE_ID}.FUNSMRY",f"{self.UNIQUE_ID}.h5",
                           f"{self.UNIQUE_ID}.MSG", f"{self.UNIQUE_ID}.PRT", f"{self.UNIQUE_ID}.PRTX",f"{self.UNIQUE_ID}.REP",
                            f"{self.UNIQUE_ID}.RTELOG", f"{self.UNIQUE_ID}.RTEMSG",f"{self.UNIQUE_ID}.ixf",f"{self.UNIQUE_ID}.bat" ]
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




        

