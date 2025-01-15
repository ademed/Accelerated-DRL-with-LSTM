
import numpy as np
import pandas as pd
import shutil
import sys, os

sys.path.append(os.getcwd())
from E2CO.Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()

from IX_Robust.Writers import  *
from IX_Robust.RunIX import *
from scipy.stats import qmc



# Class well control:
class WellControlsClass:
    def __init__(self, u, well_names, SimulationSettings):
        [rates, BHPs] = DesignExtraction(u, well_names, SimulationSettings)
        self.Q = rates
        self.BHP = BHPs



# Class LHS-sampling:
class LHSSampling:
    def __init__(self, n_data, well_names, SimulationSettings, Bounds, seed=19071996, export_csv=False, csv_folder_path=r'C:\Users\qun972\PycharmProjects\E2CORobust'):
        self.WellNames = well_names
        self.SimulationSettings = SimulationSettings
        self.Bounds = Bounds
        self.SeedNumber = seed

        self.LHSDataNormalized, self.LHSData = LHS(n_data, well_names, SimulationSettings, Bounds, seed)
        if export_csv:
            os.makedirs(csv_folder_path, exist_ok=True)
            pd.DataFrame(data=self.LHSData).to_csv(os.path.join(csv_folder_path + "\\" + "LHSData_Seed" + str(seed) + ".csv"))
            pd.DataFrame(data=self.LHSDataNormalized).to_csv(os.path.join(csv_folder_path + "\\" + "LHSDataNormalized_Seed" + str(seed) + ".csv"))

    def RunSimulation(self, IncludeLinks, ConversionFactors, DatasetFolder=r'C:\Users\qun972\PycharmProjects\E2CORobust\Dataset', SubfolderName='Realization', SubsubfolderName='Case', SimulationsToRun=None):
        n_data, n_vars = self.LHSData.shape
        NumberOfRealizations = len(IncludeLinks.BatchFile)
        ModelFolders = []
        SimulationResultFolders = []
        for Realization in range(NumberOfRealizations):
            # Get the folder paths containing simulation results
            simulationfolder = os.path.dirname(IncludeLinks.AFIDataFile[Realization])
            SimulationResultFolders.append(simulationfolder)

            # Create the target folder paths to copy simulation results over:
            datafolder = os.path.join(DatasetFolder + "\\" + SubfolderName + str(Realization + 1))
            os.makedirs(datafolder, exist_ok=True)
            ModelFolders.append(os.path.join(DatasetFolder + "\\" + SubfolderName + str(Realization + 1)))

        if SimulationsToRun is None:
            SimulationsToRun = n_data
        for i in range(SimulationsToRun):
            print("-------------------------------------------------------------------------------------------------------------")
            print("Generating results for case #" + str(i + 1) + ".....\n")
            # Run INTERSECT:
            well_controls = WellControlsClass(self.LHSData[i, :].reshape((n_vars, 1)), self.WellNames, self.SimulationSettings)
            ScheduleWriterIX(IncludeLinks, self.WellNames, well_controls, ConversionFactors, self.SimulationSettings, WriteTimeZero=False, print_notice=False)
            RunIXAll(IncludeLinks, print_simulation_cmd=False, print_notice=True, time_simulation=True)

            # Copy results over:
            for Realization in range(NumberOfRealizations):

                casefolder = os.path.join(ModelFolders[Realization] + "\\" + SubsubfolderName + str(i + 1))
                casecsvfolder = os.path.join(casefolder + "\\" + "CSV")
                casepklfolder = os.path.join(casefolder + "\\" + "PKL")
                os.makedirs(casefolder, exist_ok=True)
                os.makedirs(casecsvfolder, exist_ok=True)
                os.makedirs(casepklfolder, exist_ok=True)

                for file in os.listdir(SimulationResultFolders[Realization]):
                    if file.endswith('.csv'):
                        currentfilepath = os.path.join(SimulationResultFolders[Realization] + "\\" + file)
                        targetfilepath = os.path.join(casecsvfolder + "\\" + file)
                        shutil.copy(currentfilepath, targetfilepath)

                    if file.endswith('.pkl'):
                        currentfilepath = os.path.join(SimulationResultFolders[Realization] + "\\" + file)
                        targetfilepath = os.path.join(casepklfolder + "\\" + file)
                        shutil.copy(currentfilepath, targetfilepath)

            print("Case #" + str(i + 1) + " finished!!!\n")

    def RunSimulation_SingleRealization(self, IncludeLinks, ConversionFactors, Realization, DatasetFolder=r'C:\Users\qun972\PycharmProjects\E2CORobust\Dataset', SubfolderName='Realization', SubsubfolderName='Case',
                      SimulationsToRun=None):
        n_data, n_vars = self.LHSData.shape

        # Get the folder paths containing simulation results
        simulationfolder = os.path.dirname(IncludeLinks.AFIDataFile[Realization - 1])

        # Create the target folder paths to copy simulation results over:
        datafolder = os.path.join(DatasetFolder + "\\" + SubfolderName + str(Realization))
        os.makedirs(datafolder, exist_ok=True)
        modelfolder = os.path.join(DatasetFolder + "\\" + SubfolderName + str(Realization))

        if SimulationsToRun is None:
            SimulationsToRun = n_data
        for i in range(SimulationsToRun):
            print("-------------------------------------------------------------------------------------------------------------")
            print("Generating results for case #" + str(i + 1) + ".....\n")
            # Run INTERSECT:
            well_controls = WellControlsClass(self.LHSData[i, :].reshape((n_vars, 1)), self.WellNames, self.SimulationSettings)
            ScheduleWriterIX_Single(IncludeLinks, (Realization - 1), self.WellNames, well_controls, ConversionFactors, self.SimulationSettings, WriteTimeZero=False, print_notice=False)
            RunIX(Realization - 1, IncludeLinks, print_simulation_cmd=False, print_notice=True, time_simulation=True)
            # RunIXAll(IncludeLinks, print_simulation_cmd=False, print_notice=True, time_simulation=True)

            # Copy results over:
            casefolder = os.path.join(modelfolder + "\\" + SubsubfolderName + str(i + 1))
            casecsvfolder = os.path.join(casefolder + "\\" + "CSV")
            casepklfolder = os.path.join(casefolder + "\\" + "PKL")
            os.makedirs(casefolder, exist_ok=True)
            os.makedirs(casecsvfolder, exist_ok=True)
            os.makedirs(casepklfolder, exist_ok=True)

            for file in os.listdir(simulationfolder):
                if file.endswith('.csv'):
                    if not file.startswith('LHSData'):
                        currentfilepath = os.path.join(simulationfolder + "\\" + file)
                        targetfilepath = os.path.join(casecsvfolder + "\\" + file)
                        shutil.copy(currentfilepath, targetfilepath)

                if file.endswith('.pkl'):
                    currentfilepath = os.path.join(simulationfolder + "\\" + file)
                    targetfilepath = os.path.join(casepklfolder + "\\" + file)
                    shutil.copy(currentfilepath, targetfilepath)

            print("Case #" + str(i + 1) + " finished!!!\n")



# Function 1: Check if the 2 vectors are 1-column 2D arrays
def DimensionCheck(v1, v2):
    r1 = v1.shape[0]
    c1 = v1.shape[1]
    r2 = v2.shape[0]
    c2 = v2.shape[1]

    if min(r1, c1) != 1 or min(r2, c2) != 1:
        sys.exit(f"{bcolors.FAIL}{bcolors.BOLD}ERROR: Input Vector Dimensions Invalid!\n{bcolors.ENDC}")
    else:
        if r1 == 1:
            v1 = np.transpose(v1)

        if r2 == 1:
            v2 = np.transpose(v2)

    return [v1, v2]


# Function 2: Extract <rates> and <BHPs> vector from total well control vector <u>
def DesignExtraction(u, well_names, SimulationSettings):
    # Use .copy() to prevent function calls from changing the input variables:
    u1 = u.copy()

    Ninj = len(well_names.Injectors)
    n_cyc = SimulationSettings.NumberOfCycles
    rates = u1[0:(Ninj * n_cyc)]
    BHPs = u1[(Ninj * n_cyc):]

    return [rates, BHPs]


# Function 3: Normalize the well control vector <u> to between 0 and 1
def Normalizer(u, well_names, SimulationSettings, Bounds):
    # Use .copy() to prevent function calls from changing the input variables:
    u1 = u.copy()

    Ninj = len(well_names.Injectors)
    n_cyc = SimulationSettings.NumberOfCycles

    [rates, BHPs] = DesignExtraction(u1, well_names, SimulationSettings)

    BHPs_norm = (BHPs - Bounds.LowerBHP)/(Bounds.UpperBHP - Bounds.LowerBHP)

    rates_norm = rates
    for i in range(Ninj):
        Bound1 = i * n_cyc
        Bound2 = Bound1 + (n_cyc)           #Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
        rates_LB = np.tile(Bounds.LowerWinj, (n_cyc, 1))
        rates_UB = np.tile(Bounds.UpperWinj, (n_cyc, 1))
        rates_norm[Bound1:Bound2] = (rates[Bound1:Bound2] - rates_LB)/(rates_UB - rates_LB)

    [rates_norm, BHPs_norm] = DimensionCheck(rates_norm, BHPs_norm)
    u_norm = np.concatenate((rates_norm, BHPs_norm), axis=0)

    return u_norm


# Function 4: Denormalize the normalized well control vector <u_norm> to original <u>
def Denormalizer(u_norm, well_names, SimulationSettings, Bounds):
    # Use .copy() to prevent function calls from changing the input variables:
    u1_norm = u_norm.copy()

    Ninj = len(well_names.Injectors)
    n_cyc = SimulationSettings.NumberOfCycles

    [rates_norm, BHPs_norm] = DesignExtraction(u1_norm, well_names, SimulationSettings)

    BHPs = BHPs_norm*(Bounds.UpperBHP - Bounds.LowerBHP) + Bounds.LowerBHP

    rates = rates_norm
    for i in range(Ninj):
        Bound1 = i * n_cyc
        Bound2 = Bound1 + (n_cyc)           #Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
        rates_LB = np.tile(Bounds.LowerWinj, (n_cyc, 1))
        rates_UB = np.tile(Bounds.UpperWinj, (n_cyc, 1))
        rates[Bound1:Bound2] = rates_norm[Bound1:Bound2]*(rates_UB - rates_LB) + rates_LB

    [rates, BHPs] = DimensionCheck(rates, BHPs)
    u_orig = np.concatenate((rates, BHPs), axis=0)

    return u_orig


# Function 5: Latin Hypercube Sampling (LHS) between the bounds for the well controls:
def LHS(n_data, well_names, SimulationSettings, Bounds, seed=19071996):
    Ninj = len(well_names.Injectors)
    Nprd = len(well_names.Producers)
    n_cyc = SimulationSettings.NumberOfCycles
    n_vars = (Ninj + Nprd)*n_cyc
    if seed != 0:
        LHSOperator = qmc.LatinHypercube(d=n_vars, seed=int(abs(seed)))
    else:
        LHSOperator = qmc.LatinHypercube(d=n_vars)

    u_LHS_norm = LHSOperator.random(n=n_data)
    u_LHS = np.zeros(u_LHS_norm.shape)

    for i in range(u_LHS.shape[0]):
        u_LHS[i, :] = Denormalizer(u_LHS_norm[i, :].reshape((n_vars, 1)), well_names, SimulationSettings, Bounds).ravel()

    return u_LHS_norm, u_LHS



def TimeAverage(data, TimeVector):
    data_filtered = data[data['TIME'].isin(TimeVector[1:])].loc[1:, :]
    colnames = data.columns
    for i in range(1, len(TimeVector)):
        active_df = data.loc[(TimeVector[i-1] <= data['TIME']) & (data['TIME'] <= TimeVector[i])]
        n = active_df.shape[0]
        time_difference = active_df['TIME'].iloc[1:n].to_numpy() - active_df['TIME'].iloc[0:(n-1)].to_numpy()
        for col in colnames:
            if col != 'TIME':
                data_filtered[col].iloc[i - 1] = np.sum((active_df[col].iloc[1:n].to_numpy() + active_df[col].iloc[0:(n-1)].to_numpy()) * time_difference)/2/(TimeVector[i] - TimeVector[i-1])

    return data_filtered