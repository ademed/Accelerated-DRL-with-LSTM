import numpy as np
import os

from E2CO.Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()


class ConversionFactors:
    """
    Class attributes:
        ConversionFactors.Rate_Field_to_ModSI: to convert flowrates from field unit (stb/d) to Modified SI unit.
        ConversionFactors.Pressure_Field_to_ModSI: to convert pressure from field unit (psi) to Modified SI unit.
    """

    def __init__(self):
        self.Rate_Field_to_ModSI = 0.1589873
        self.Pressure_Field_to_ModSI = 0.0703069578


class SimulationSettings:
    """
    Inputs:
        INUNITS: Input unit (INUNITS), defined at the beginning of the CMG simulation datafile (default = "MODSI")
        OUTUNITS: Output unit (OUTUNITS), defined at the beginning of the CMG simulation datafile (default = "Field")
        TotalLife: Total simulation lifetime in days (or cycle lifetime, default = 3600)
        NumberOfCycles: Number of cycles (or control steps, default = 20)
        NumberOfSimulationSteps: Number of steps in between one control step to request outputs (default = 5)
        TimeZero: Time (in days) where the simulation begins (default = 0)

    Class attributes:
        SimulationSettings.SimulationINUNITS = Input unit (INUNITS), defined at the beginning of the CMG simulation datafile.
        SimulationSettings.SimulationOUTUNITS = Output unit (OUTUNITS), defined at the beginning of the CMG simulation datafile.
        SimulationSettings.NumberOfCycles = Number of cycles (or control steps).
        SimulationSettings.TotalLife = Total simulation lifetime in days.
        SimulationSettings.NumberOfSimulationSteps = Number of steps in between one control step to request outputs.
        SimulationSettings.TimeZero = Time (in days) where the simulation begins.
        SimulationSettings.ControlStep = Control step length (in days).
        SimulationSettings.SimulationTimeStep = Simulation export time step length (in days).
    """
    def __init__(self, INUNITS="MODSI", OUTUNITS="Field", TotalLife=3600, NumberOfCycles=20, NumberOfSimulationSteps=5, TimeZero=0):
        self.SimulationINUNITS = INUNITS
        self.SimulationOUTUNITS = OUTUNITS
        self.NumberOfCycles = NumberOfCycles
        self.TotalLife = TotalLife
        self.NumberOfSimulationSteps = NumberOfSimulationSteps
        self.TimeZero = TimeZero
        self.ControlStep = self.TotalLife / self.NumberOfCycles
        self.SimulationTimeStep = self.ControlStep / self.NumberOfSimulationSteps


class NumericalSettings:
    """
    Inputs:
        JACPAR: Option (in boolean) to run CMG in parallel (using multiple CPU cores, default = True).
        DTMAX: DTMAX value in NUMERICAL section of CMG datafile (default = 30).
        ITERMAX: ITERMAX value in NUMERICAL section of CMG datafile (default = 200).
        NCUTS: NCUTS value in NUMERICAL section of CMG datafile (default = 10).
        MaxChangePressure: MAXCHANGE PRESS value in NUMERICAL section of CMG datafile (default = 5).
        MaxChangeSaturation: MAXCHANGE SATUR value in NUMERICAL section of CMG datafile (default = 0.1).
        NormPress: NORM PRESS value in NUMERICAL section of CMG datafile (default = 1).
        NormSaturation: NORM SATUR value in NUMERICAL section of CMG datafile (default = 0.05).

    Class attributes:
        self.MaxCPUCores: Maximum number of CPU cores available on user's computer (self-detected).
        self.MaxChangePressure = MAXCHANGE PRESS value in NUMERICAL section of CMG datafile.
        self.MaxChangeSaturation = MAXCHANGE SATUR value in NUMERICAL section of CMG datafile.
        self.NormPress = NORM PRESS value in NUMERICAL section of CMG datafile.
        self.NormSaturation = NORM SATUR value in NUMERICAL section of CMG datafile.
        self.NCUTS = NCUTS value in NUMERICAL section of CMG datafile.
        self.DTMAX = DTMAX value in NUMERICAL section of CMG datafile.
        self.ITERMAX = ITERMAX value in NUMERICAL section of CMG datafile.
            self.JacPar = "ON" if JACPAR = True, "OFF" is JACPAR = False.
            self.Solver = "PARASOL" if JACPAR = True, "AIMSOL" is JACPAR = False.

    """

    def __init__(self, JACPAR=True, DTMAX=30, ITERMAX=200, NCUTS=10, MaxChangePressure=5, MaxChangeSaturation=0.1, NormPress=1, NormSaturation=0.05, MAXSTEPS=100000):
        self.MaxCPUCores = os.cpu_count()
        self.MaxChangePressure = MaxChangePressure
        self.MaxChangeSaturation = MaxChangeSaturation
        self.NormPress = NormPress
        self.NormSaturation = NormSaturation
        self.NCUTS = NCUTS
        self.DTMAX = DTMAX
        self.ITERMAX = ITERMAX
        self.MAXSTEPS = MAXSTEPS
        if JACPAR:
            self.JacPar = "ON"
            self.Solver = "PARASOL"
        else:
            self.JacPar = "OFF"
            self.Solver = "AIMSOL"


class RwdSettings:
    def __init__(self, InjectorExport="Groups", ProducerExport="Groups", InjectorsList="N/A", ProducersList="N/A",
                 InjectorParameters=["FWIT"],
                 ProducerParameters=["FOPT", "FWPT", "FLPT", "FGPT"]):
        self.InjectorExport = InjectorExport
        self.ProducerExport = ProducerExport
        self.Injectors = InjectorsList
        self.Producers = ProducersList
        self.InjectorsParameters = InjectorParameters
        self.ProducersParameters = ProducerParameters
        self.InjectorsParameterKeywords = []
        self.ProducersParameterKeywords = []
        for p in self.InjectorsParameters:
            if p == "FWIT":
                self.InjectorsParameterKeywords.append('WATER_INJECTION_CUML')
            elif p == "FLIT":
                self.InjectorsParameterKeywords.append('LIQUID_INJECTION_CUML')
            elif p == "FGIT":
                self.InjectorsParameterKeywords.append('GAS_INJECTION_CUML')

        for p in self.ProducersParameters:
            if p == "FOPT":
                self.ProducersParameterKeywords.append('OIL_PRODUCTION_CUML')
            elif p == "FWPT":
                self.ProducersParameterKeywords.append('WATER_PRODUCTION_CUML')
            elif p == "FLPT":
                self.ProducersParameterKeywords.append('LIQUID_PRODUCTION_CUML')
            elif p == "FGPT":
                self.ProducersParameterKeywords.append('GAS_PRODUCTION_CUML')



# Function 1: CMG Numerical Settings
def NumericalWriter(IncludeLinks, NumericalSettings, print_notice=False, CPUThreshold=6, SpareCPU=4):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    print_multiple = True
    for i in range(NumberOfRealizations):
        fileID = open(IncludeLinks.Numerical[i], "w+")
        ####################################################################################################################
        # Begin writing:
        fileID.write("NUMERICAL\n\n")

        ####################################################################################################################
        # Middle stuffs (DTMAX, MAXSTEPS and P, Sw settings):
        fileID.write("DTMAX " + str(NumericalSettings.DTMAX) + "\n")
        fileID.write("MAXCHANGE PRESS " + str(NumericalSettings.MaxChangePressure) + "\n")
        fileID.write("MAXCHANGE SATUR " + str(NumericalSettings.MaxChangeSaturation) + "\n")
        fileID.write("NORM PRESS " + str(NumericalSettings.NormPress) + "\n")
        fileID.write("NORM SATUR " + str(NumericalSettings.NormSaturation) + "\n")

        ####################################################################################################################
        # Newton cuts (NCUTS) and max iterations (ITERMAX):
        fileID.write("NCUTS " + str(NumericalSettings.NCUTS) + "\n\n")
        fileID.write("ITERMAX " + str(NumericalSettings.ITERMAX) + "\n\n")

        ####################################################################################################################
        # Parallel settings for CMG (JACPAR):
        if NumericalSettings.JacPar == "ON":
            fileID.write("JACPAR " + NumericalSettings.JacPar + "\n")
            fileID.write("SOLVER " + NumericalSettings.Solver + "\n")

            if print_multiple:
                print(f"{bcolors.HEADER}************************************************************************{bcolors.ENDC}")
                print(f"{bcolors.HEADER}{bcolors.BOLD}NUMERICAL SETTINGS NOTICE: {bcolors.ENDC}")

            if NumericalSettings.MaxCPUCores <= CPUThreshold:
                CPUCores = 2
                if print_multiple:
                    print(f"      {bcolors.WARNING}WARNING!{bcolors.ENDC} YOUR COMPUTER IS PRETTY TRASH WITH ONLY " +
                          str(NumericalSettings.MaxCPUCores) + " CPU CORES! CONSIDER REPLACE!!")
                    print(
                        "      BECAUSE OF LIMITED CPU RESOURCES, ONLY " + str(CPUCores) + " ARE USED FOR PARALLEL SIMULATION!")
            else:
                if print_multiple:
                    print("      WOW! PRETTY GOOD COMPUTER WITH " + str(NumericalSettings.MaxCPUCores) + " CPU CORES!!!!")
                    CPUCores = NumericalSettings.MaxCPUCores - SpareCPU
                    print("      NUMBER OF CPU CORES USED FOR PARALLEL SIMULATION: " + str(CPUCores) + ".")

            if print_multiple:
                print(
                    f"      ({bcolors.WARNING}{bcolors.BOLD}{bcolors.UNDERLINE}NOTE:{bcolors.ENDC} This number of CPU cores used can be modified in {bcolors.BOLD}{bcolors.OKCYAN}NumericalWrite"
                    f"r{bcolors.ENDC} function located in {bcolors.BOLD}{bcolors.OKBLUE}.\CMG\Writers.py{bcolors.ENDC})")
                print(f"{bcolors.HEADER}************************************************************************{bcolors.ENDC}")
                print_multiple = False

            fileID.write("PNTHRDS " + str(CPUCores) + "\n")
            fileID.write("PPATTERN AUTOPSLAB " + str(CPUCores) + "\n")

        fileID.close()
    ####################################################################################################################
    # Print out completion notice if requested:
    if print_notice:
        print(
            f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.BOLD}{bcolors.OKGREEN}Simulation {bcolors.UNDERLINE}NUMERICAL{bcolors.ENDC} {bcolors.BOLD}"
                                                                                  f"{bcolors.OKGREEN}settings files are generated for ALL realizations!\n{bcolors.ENDC}")

    return None


# Function 2: IX Schedule Writer
def ScheduleWriterIX(IncludeLinks, well_names, well_controls, ConversionFactors, SimulationSettings, WriteTimeZero=False,
                   print_notice=False):
    ####################################################################################################################
    # Pre-process by converting units from FIELD (optimization - OUTUNITS) to ModSI (simulation - INUNITS):
    if SimulationSettings.SimulationINUNITS != SimulationSettings.SimulationOUTUNITS:
        if SimulationSettings.SimulationOUTUNITS == "Field":
            if SimulationSettings.SimulationINUNITS == "MODSI":
                Q_vec = well_controls.Q * ConversionFactors.Rate_Field_to_ModSI
                BHP_vec = well_controls.BHP * ConversionFactors.Pressure_Field_to_ModSI
            else:
                Q_vec = well_controls.Q * ConversionFactors.Rate_Field_to_SI
                BHP_vec = well_controls.BHP * ConversionFactors.Pressure_Field_to_SI
        elif SimulationSettings.SimulationOUTUNITS == "SI":
            if SimulationSettings.SimulationINUNITS == "Field":
                Q_vec = well_controls.Q / ConversionFactors.Rate_Field_to_SI
                BHP_vec = well_controls.BHP / ConversionFactors.Pressure_Field_to_SI
        else:
            if SimulationSettings.SimulationINUNITS == "Field":
                Q_vec = well_controls.Q / ConversionFactors.Rate_Field_to_ModSI
                BHP_vec = well_controls.BHP / ConversionFactors.Pressure_Field_to_ModSI
    else:
        Q_vec = well_controls.Q
        BHP_vec = well_controls.BHP

    ####################################################################################################################
    Ninj = len(well_names.Injectors)
    Nprd = len(well_names.Producers)
    t0 = SimulationSettings.TimeZero
    n_cyc = SimulationSettings.NumberOfCycles

    NumberOfRealizations = len(IncludeLinks.BatchFile)
    for n_r in range(NumberOfRealizations):
        fileID = open(IncludeLinks.Schedule[n_r], "w+")
        time = t0
        ####################################################################################################################
        # Start Writing:
        fileID.write("MODEL_DEFINITION\n\n\n\n")
        fileID.write("START\n")

        if t0 == 0:
            if not WriteTimeZero:
                fileID.write("##TIME " + str(t0) + "\n")
            else:
                fileID.write("TIME " + str(t0) + "\n")
        fileID.write("\n")

        fileID.write("StaticList \"I\\*\" {\n")
        fileID.write("    EntityList=[ Well(")
        for j in range(Ninj):
            fileID.write(" \"" + well_names.Injectors[j] + "\"")
        fileID.write(") ]\n")
        fileID.write("}\n\n")

        fileID.write("StaticList \"P\\*\" {\n")
        fileID.write("    EntityList=[ Well(")
        for k in range(Nprd):
            fileID.write(" \"" + well_names.Producers[k] + "\"")
        fileID.write(") ]\n")
        fileID.write("}\n\n")

        fileID.write("Group \"" + well_names.GroupName + "\" {\n")
        fileID.write("    Members=[ Well(")
        for j in range(Ninj):
            fileID.write(" \"" + well_names.Injectors[j] + "\"")
        for k in range(Nprd):
            fileID.write(" \"" + well_names.Producers[k] + "\"")
        fileID.write(") ]\n")
        fileID.write("}\n\n")

        ####################################################################################################################
        # Creating big matrices for ease of calling:
        Q_mat = np.zeros([n_cyc, Ninj])
        BHP_mat = np.zeros([n_cyc, Nprd])
        for j in range(Ninj):
            Bound1 = j * n_cyc
            Bound2 = Bound1 + (n_cyc)  # Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
            Q_mat[:, j] = Q_vec[Bound1:Bound2].ravel()

        for j in range(Nprd):
            Bound1 = j * n_cyc
            Bound2 = Bound1 + (n_cyc)  # Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
            BHP_mat[:, j] = BHP_vec[Bound1:Bound2].ravel()

        fileID.write("\n")

        ####################################################################################################################
        # Writing the well controls for each cycle:
        for i in range(n_cyc):
            # Starting Water/Solvent Injection:
            fileID.write("################################################\n")
            fileID.write("## CONTROL STEP #" + str(i + 1) + ": \n")
            fileID.write("################################################\n\n")

            for j in range(Ninj):
                if i == 0:
                    fileID.write("Well \"" + well_names.Injectors[j] + "\" {\n")
                    fileID.write("    Status=OPEN\n")
                    fileID.write("    Type=WATER_INJECTOR\n")
                    fileID.write("    Constraints=[ Constraint(" + str(Q_mat[i, j]) + " WATER_INJECTION_RATE)]\n")
                    fileID.write("    HonorInjectionStreamAvailability = FALSE\n")
                    fileID.write("}\n\n")
                else:
                    # fileID.write("Well(\'" + well_names.Injectors[j] + "\').set_constraint((" + str(Q_mat[i, j]) + ", WATER_INJECTION_RATE))\n\n")
                    fileID.write("Well \"" + well_names.Injectors[j] + "\" {\n")
                    fileID.write("    Status=OPEN\n")
                    fileID.write("    Type=WATER_INJECTOR\n")
                    fileID.write("    remove_all_constraints( )\n")
                    fileID.write("    Constraints=[ Constraint(" + str(Q_mat[i, j]) + " WATER_INJECTION_RATE)]\n")
                    fileID.write("    HonorInjectionStreamAvailability = FALSE\n")
                    fileID.write("}\n\n")
            for k in range(Nprd):
                if i == 0:
                    fileID.write("Well \"" + well_names.Producers[k] + "\" {\n")
                    fileID.write("    Status=OPEN\n")
                    fileID.write("    Type=PRODUCER\n")
                    fileID.write("    Constraints=[ Constraint(" + str(BHP_mat[i, k]) + " BOTTOM_HOLE_PRESSURE)]\n")
                    fileID.write("}\n")
                else:
                    # fileID.write("Well(\'" + well_names.Producers[k] + "\').set_constraint((" + str(BHP_mat[i, k]) + ", BOTTOM_HOLE_PRESSURE))\n\n")
                    fileID.write("Well \"" + well_names.Producers[k] + "\" {\n")
                    fileID.write("    Status=OPEN\n")
                    fileID.write("    Type=PRODUCER\n")
                    fileID.write("    remove_all_constraints( )\n")
                    fileID.write("    Constraints=[ Constraint(" + str(BHP_mat[i, k]) + " BOTTOM_HOLE_PRESSURE)]\n")
                    fileID.write("}\n")

            fileID.write("\n")

            for k in range(SimulationSettings.NumberOfSimulationSteps):
                time = time + SimulationSettings.SimulationTimeStep
                fileID.write("TIME " + str("{:.4f}".format(time)) + "\n")

            fileID.write("\n")

        ####################################################################################################################
        fileID.write("END_INPUT\n\n")
        fileID.close()
    # Print out completion notice if requested:
    if print_notice:
        print(
            f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.BOLD}{bcolors.OKGREEN}Simulation/Optimization {bcolors.UNDERLINE}SCHEDULE{bcolors.ENDC} {bcolors.BOLD}"
                                                                                  f"{bcolors.OKGREEN}files are generated for all realizations!\n{bcolors.ENDC}")

    return None

def ScheduleWriterIX_Single(IncludeLinks, Realization, well_names, well_controls, ConversionFactors, SimulationSettings, WriteTimeZero=False,
                   print_notice=False):
    ####################################################################################################################
    # Pre-process by converting units from FIELD (optimization - OUTUNITS) to ModSI (simulation - INUNITS):
    if SimulationSettings.SimulationINUNITS != SimulationSettings.SimulationOUTUNITS:
        if SimulationSettings.SimulationOUTUNITS == "Field":
            if SimulationSettings.SimulationINUNITS == "MODSI":
                Q_vec = well_controls.Q * ConversionFactors.Rate_Field_to_ModSI
                BHP_vec = well_controls.BHP * ConversionFactors.Pressure_Field_to_ModSI
            else:
                Q_vec = well_controls.Q * ConversionFactors.Rate_Field_to_SI
                BHP_vec = well_controls.BHP * ConversionFactors.Pressure_Field_to_SI
        elif SimulationSettings.SimulationOUTUNITS == "SI":
            if SimulationSettings.SimulationINUNITS == "Field":
                Q_vec = well_controls.Q / ConversionFactors.Rate_Field_to_SI
                BHP_vec = well_controls.BHP / ConversionFactors.Pressure_Field_to_SI
        else:
            if SimulationSettings.SimulationINUNITS == "Field":
                Q_vec = well_controls.Q / ConversionFactors.Rate_Field_to_ModSI
                BHP_vec = well_controls.BHP / ConversionFactors.Pressure_Field_to_ModSI
    else:
        Q_vec = well_controls.Q
        BHP_vec = well_controls.BHP

    ####################################################################################################################
    Ninj = len(well_names.Injectors)
    Nprd = len(well_names.Producers)
    t0 = SimulationSettings.TimeZero
    n_cyc = SimulationSettings.NumberOfCycles

    fileID = open(IncludeLinks.Schedule[Realization], "w+")
    time = t0
    ####################################################################################################################
    # Start Writing:
    fileID.write("MODEL_DEFINITION\n\n\n\n")
    fileID.write("START\n")

    if t0 == 0:
        if not WriteTimeZero:
            fileID.write("##TIME " + str(t0) + "\n")
        else:
            fileID.write("TIME " + str(t0) + "\n")
    fileID.write("\n")

    fileID.write("StaticList \"I\\*\" {\n")
    fileID.write("    EntityList=[ Well(")
    for j in range(Ninj):
        fileID.write(" \"" + well_names.Injectors[j] + "\"")
    fileID.write(") ]\n")
    fileID.write("}\n\n")

    fileID.write("StaticList \"P\\*\" {\n")
    fileID.write("    EntityList=[ Well(")
    for k in range(Nprd):
        fileID.write(" \"" + well_names.Producers[k] + "\"")
    fileID.write(") ]\n")
    fileID.write("}\n\n")

    fileID.write("Group \"" + well_names.GroupName + "\" {\n")
    fileID.write("    Members=[ Well(")
    for j in range(Ninj):
        fileID.write(" \"" + well_names.Injectors[j] + "\"")
    for k in range(Nprd):
        fileID.write(" \"" + well_names.Producers[k] + "\"")
    fileID.write(") ]\n")
    fileID.write("}\n\n")

    ####################################################################################################################
    # Creating big matrices for ease of calling:
    Q_mat = np.zeros([n_cyc, Ninj])
    BHP_mat = np.zeros([n_cyc, Nprd])
    for j in range(Ninj):
        Bound1 = j * n_cyc
        Bound2 = Bound1 + (n_cyc)  # Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
        Q_mat[:, j] = Q_vec[Bound1:Bound2].ravel()

    for j in range(Nprd):
        Bound1 = j * n_cyc
        Bound2 = Bound1 + (n_cyc)  # Does not have (n_cyc - 1) due to how numpy slicing/indexing works.
        BHP_mat[:, j] = BHP_vec[Bound1:Bound2].ravel()

    fileID.write("\n")

    ####################################################################################################################
    # Writing the well controls for each cycle:
    for i in range(n_cyc):
        # Starting Water/Solvent Injection:
        fileID.write("################################################\n")
        fileID.write("## CONTROL STEP #" + str(i + 1) + ": \n")
        fileID.write("################################################\n\n")

        for j in range(Ninj):
            if i == 0:
                fileID.write("Well \"" + well_names.Injectors[j] + "\" {\n")
                fileID.write("    Status=OPEN\n")
                fileID.write("    Type=WATER_INJECTOR\n")
                fileID.write("    Constraints=[ Constraint(" + str(Q_mat[i, j]) + " WATER_INJECTION_RATE)]\n")
                fileID.write("    HonorInjectionStreamAvailability = FALSE\n")
                fileID.write("}\n\n")
            else:
                # fileID.write("Well(\'" + well_names.Injectors[j] + "\').set_constraint((" + str(Q_mat[i, j]) + ", WATER_INJECTION_RATE))\n\n")
                fileID.write("Well \"" + well_names.Injectors[j] + "\" {\n")
                fileID.write("    Status=OPEN\n")
                fileID.write("    Type=WATER_INJECTOR\n")
                fileID.write("    remove_all_constraints( )\n")
                fileID.write("    Constraints=[ Constraint(" + str(Q_mat[i, j]) + " WATER_INJECTION_RATE)]\n")
                fileID.write("    HonorInjectionStreamAvailability = FALSE\n")
                fileID.write("}\n\n")
        for k in range(Nprd):
            if i == 0:
                fileID.write("Well \"" + well_names.Producers[k] + "\" {\n")
                fileID.write("    Status=OPEN\n")
                fileID.write("    Type=PRODUCER\n")
                fileID.write("    Constraints=[ Constraint(" + str(BHP_mat[i, k]) + " BOTTOM_HOLE_PRESSURE)]\n")
                fileID.write("}\n")
            else:
                # fileID.write("Well(\'" + well_names.Producers[k] + "\').set_constraint((" + str(BHP_mat[i, k]) + ", BOTTOM_HOLE_PRESSURE))\n\n")
                fileID.write("Well \"" + well_names.Producers[k] + "\" {\n")
                fileID.write("    Status=OPEN\n")
                fileID.write("    Type=PRODUCER\n")
                fileID.write("    remove_all_constraints( )\n")
                fileID.write("    Constraints=[ Constraint(" + str(BHP_mat[i, k]) + " BOTTOM_HOLE_PRESSURE)]\n")
                fileID.write("}\n")

        fileID.write("\n")

        for k in range(SimulationSettings.NumberOfSimulationSteps):
            time = time + SimulationSettings.SimulationTimeStep
            fileID.write("TIME " + str("{:.4f}".format(time)) + "\n")

        fileID.write("\n")

    ####################################################################################################################
    fileID.write("END_INPUT\n\n")
    fileID.close()

    # Print out completion notice if requested:
    if print_notice:
        print(
            f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.BOLD}{bcolors.OKGREEN}Simulation/Optimization {bcolors.UNDERLINE}SCHEDULE{bcolors.ENDC} {bcolors.BOLD}"
                                                                                  f"{bcolors.OKGREEN}files are generated for all realizations!\n{bcolors.ENDC}")

    return None

### THESE FUNCTIONS ARE TO WRITE BUILT-IN PYTHON SCRIPTS IN IX .afi FILE TO EXTRACT NONLINEAR CONSTRAINT INFORMATION
# Function 3: SINGLE custom script writer:
def CustomControlWriter(CustomScriptsfileID, filename, Parameter, ParameterKeyword, GroupName):
    fileID = CustomScriptsfileID
    fileID.write("CustomControl \"Export" + Parameter + "Data\" {\n")

    fileID.write("	ExecutionPosition=BEGIN_TIMESTEP")
    fileID.write("\n\n")

    # Write the InitialScript:
    fileID.write("	InitialScript=@{\n")
    fileID.write("filename = \'" + filename + '\'' + '\n')
    fileID.write("_file = open(filename, \"w\")" + '\n')
    fileID.write("_current_wells = []" + '\n')
    fileID.write("for well in	Group(\'" + GroupName + "\').FlowEntities.get():" + '\n')
    fileID.write("	if well not in _current_wells:\n")
    fileID.write("		_current_wells.append(well)\n")
    fileID.write("print(\"DATE, TIME\", end=\'\', file=_file)\n")
    fileID.write("for well in _current_wells:\n")
    fileID.write("	print(\',\', well.get_name(), end=\'\', file=_file)\n")
    fileID.write("print(\'\\n\', end=\'\', file=_file)\n")
    fileID.write("}@\n\n")

    # Write the main Script:
    fileID.write("	Script=@{\n")
    fileID.write("print(FieldManagement().CurrentDate.get(), end=\'\', file=_file)\n")
    fileID.write("print(\',\', FieldManagement().CurrentTime.get(), end=\'\', file=_file)\n\n")

    fileID.write("for well in _current_wells:\n")
    fileID.write("	print(\',\', Well(well.get_name()).get_property("+ ParameterKeyword + ").value, end=\'\', file=_file)\n\n")

    fileID.write("print(\'\\n\', end=\'\', file=_file)\n")
    fileID.write("}@\n\n")

    # Write the final script:
    fileID.write("	FinalScript=@{\n")
    fileID.write("print(FieldManagement().CurrentDate.get(), end=\'\', file=_file)\n")
    fileID.write("print(\',\', FieldManagement().CurrentTime.get(), end=\'\', file=_file)\n\n")

    fileID.write("for well in _current_wells:\n")
    fileID.write("	print(\',\', Well(well.get_name()).get_property(" + ParameterKeyword + ").value, end=\'\', file=_file)\n\n")

    fileID.write("_file.close()\n")
    fileID.write("print(\"File\", filename, \"is written for " + Parameter + " data.\")\n")
    fileID.write("}@\n\n")

    fileID.write("}\n")

    return None

# Function 4: Custom scripts writer for nonlinear constraints for all realizations:
def CustomScriptsWriter(IncludeLinks, NonlinearConstraintsSettings, GroupName='G1'):

    NumberOfRealizations = len(IncludeLinks.BatchFile)
    for Realization in range(NumberOfRealizations):

        fileID = open(IncludeLinks.CustomScripts[Realization], "w")
        fileID.write("MODEL_DEFINITION\n\n\n")

        if NonlinearConstraintsSettings.FLPR:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi","") + "_WLPT.csv"
            CustomControlWriter(fileID, filename, Parameter='WLPT', ParameterKeyword='LIQUID_PRODUCTION_CUML', GroupName=GroupName)
            fileID.write("\n\n\n")

        if NonlinearConstraintsSettings.FWPR:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi","") + "_WWPT.csv"
            CustomControlWriter(fileID, filename, Parameter='WWPT', ParameterKeyword='WATER_PRODUCTION_CUML', GroupName=GroupName)
            fileID.write("\n\n\n")

        if NonlinearConstraintsSettings.FOPR:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi", "") + "_WOPT.csv"
            CustomControlWriter(fileID, filename, Parameter='WOPT', ParameterKeyword='OIL_PRODUCTION_CUML', GroupName=GroupName)
            fileID.write("\n\n\n")

            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi", "") + "_WOPR.csv"
            CustomControlWriter(fileID, filename, Parameter='WOPR', ParameterKeyword='OIL_PRODUCTION_RATE', GroupName=GroupName)
            fileID.write("\n\n\n")

        if NonlinearConstraintsSettings.FGPR:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi","") + "_WGPT.csv"
            CustomControlWriter(fileID, filename, Parameter='WGPT', ParameterKeyword='GAS_PRODUCTION_CUML', GroupName=GroupName)
            fileID.write("\n\n\n")

        if NonlinearConstraintsSettings.FWIR:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi","") + "_WWIT.csv"
            CustomControlWriter(fileID, filename, Parameter='WWIT', ParameterKeyword='WATER_INJECTION_CUML', GroupName=GroupName)
            fileID.write("\n\n\n")

        if NonlinearConstraintsSettings.IBHP or NonlinearConstraintsSettings.WBHP:
            filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi", "") + "_WBHP.csv"
            CustomControlWriter(fileID, filename, Parameter='WBHP', ParameterKeyword='BOTTOM_HOLE_PRESSURE', GroupName=GroupName)
            fileID.write("\n\n\n")

        fileID.close()

    return None

### THESE FUNCTIONS ARE TO WRITE BUILT-IN PYTHON SCRIPTS IN IX .afi FILE TO EXTRACT INFORMATION FOR NPV
# Function 5: SINGLE custom script writer for NPV data for all realizations:
def CustomControlNPVWriter(IncludeLinks, RwdSettings, GroupName):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    for Realization in range(NumberOfRealizations):
        filename = os.path.split(IncludeLinks.AFIDataFile[Realization])[1].replace(".afi","") + "_NPVData.csv"

        fileID = open(IncludeLinks.CustomScriptsForNPV[Realization], "w")
        fileID.write("MODEL_DEFINITION\n\n\n")
        fileID.write("CustomControl \"ExportDataForNPV\" {\n")

        fileID.write("	ExecutionPosition=BEGIN_TIMESTEP")
        fileID.write("\n\n")

        # Write the InitialScript:
        fileID.write("	InitialScript=@{\n")
        fileID.write("filename = \'" + filename + '\'' + '\n')
        fileID.write("_file = open(filename, \"w\")" + '\n')

        _parameters = []
        _parameter_keys = []
        for parameter in RwdSettings.InjectorsParameters:
            _parameters.append(parameter)
        for parameter in RwdSettings.ProducersParameters:
            _parameters.append(parameter)
        for key in RwdSettings.InjectorsParameterKeywords:
            _parameter_keys.append(key)
        for key in RwdSettings.ProducersParameterKeywords:
            _parameter_keys.append(key)

        fileID.write("_parameters = " + str(_parameters) + "\n")
        fileID.write("_parameter_keys = " + str(_parameter_keys) + "\n")
        fileID.write("print(\"DATE, TIME\", end=\'\', file=_file)\n")
        fileID.write("for parameter in _parameters:" + '\n')
        fileID.write("	print(\',\', parameter, end=\'\', file=_file)\n")
        fileID.write("print(\'\\n\', end=\'\', file=_file)\n")
        fileID.write("}@\n\n")

        # Write the main Script:
        fileID.write("	Script=@{\n")
        fileID.write("print(FieldManagement().CurrentDate.get(), end=\'\', file=_file)\n")
        fileID.write("print(\',\', FieldManagement().CurrentTime.get(), end=\'\', file=_file)\n\n")

        for key in _parameter_keys:
            fileID.write("print(\',\', Group(\'"+ GroupName + "\').get_property(" + key + ").value, end=\'\', file=_file)\n")
        fileID.write("\n")
        fileID.write("print(\'\\n\', end=\'\', file=_file)\n")
        fileID.write("}@\n\n")

        # Write the final script:
        fileID.write("	FinalScript=@{\n")
        fileID.write("print(FieldManagement().CurrentDate.get(), end=\'\', file=_file)\n")
        fileID.write("print(\',\', FieldManagement().CurrentTime.get(), end=\'\', file=_file)\n\n")

        for key in _parameter_keys:
            fileID.write("print(\',\', Group(\'"+ GroupName + "\').get_property(" + key + ").value, end=\'\', file=_file)\n")

        fileID.write("\n")
        fileID.write("_file.close()\n")
        fileID.write("print(\"File\", filename, \"is written for NPV data.\")\n")
        fileID.write("}@\n\n")

        fileID.write("}\n")
        fileID.close()

    return None


### THESE FUNCTIONS ARE TO WRITE BUILT-IN PYTHON SCRIPTS IN IX .afi FILE TO EXTRACT INFORMATION GRID-SNAPSHOTS OF PRESSURE AND WATER SATURATION
def CustomControlPS_PKL(CustomScriptsfileID, SimulationSettings, naming_mode='control-step', StructuredInfoGrid='CoarseGrid', SWAT=True, SOIL=False):

    fileID = open(CustomScriptsfileID, "w")
    fileID.write("MODEL_DEFINITION\n\n\n")

    fileID.write("################################################################################################################\n")
    fileID.write("# THIS SCRIPT EXTRACT PRESSURE AND WATER SATURATION INFORMATION OF THE GRIDBLOCKS AND EXPORT IT INTO PICKLE  (.PKL) FILES \n \n")
    fileID.write("################################################################################################################\n")

    fileID.write("CustomControl \"GRID_PRESSURE_SWAT\" {\n")
    fileID.write("	ExecutionPosition=END_TIMESTEP")
    fileID.write("\n\n")

    #################################################################################################################
    # Write the InitialScript:
    fileID.write("	InitialScript = @{\n")
    fileID.write("import pickle" + '\n')
    fileID.write("import numpy as np" + '\n')
    fileID.write("_basefilename = Reservoir().get_name() " + "+ \"_Report\"" + '\n')
    fileID.write("extension = \".pkl\"" + '\n')
    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the main Script:
    fileID.write("	Script=@{\n")

    #Function Definition:
    fileID.write("def report(rTime):\n")
    fileID.write("	allData = []\n")
    fileID.write("	fCellId = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").FirstCellId.get()\n")
    fileID.write("	nX = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInI.get()\n")
    fileID.write("	nY = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInJ.get()\n")
    fileID.write("	nZ = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInK.get()\n")
    fileID.write("	cSelectionName = 'MyReservoir'\n")
    fileID.write("	selectionName = 'ResCells'\n")
    fileID.write("	cellsSelected = []\n")
    fileID.write("	Data = []\n")
    fileID.write("	cI = []\n")
    fileID.write("	cJ = []\n")
    fileID.write("	cK = []\n")
    fileID.write("	for k in range(nZ):\n")
    fileID.write("		for j in range(nY):\n")
    fileID.write("			for i in range(nX):\n")
    fileID.write("				cellId = fCellId + (i) + (j)*nX + (k)*nX*nY\n")
    fileID.write("				cellsSelected.append(cellId)\n")
    fileID.write("				cI.append(i + 1)\n")
    fileID.write("				cJ.append(j + 1)\n")
    fileID.write("				cK.append(k + 1)\n")
    fileID.write("	CellSelectionFamily(cSelectionName).add_selection(selectionName, cellsSelected)\n")
    fileID.write("	cID = GridMgr('GridMgr').RegionFamilyMgr('RegionFamilyMgr').CellSelectionFamily(cSelectionName).Cells.value\n")
    fileID.write("	cPress = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,PRESSURE).value\n")

    if SWAT:
        fileID.write("	cSWAT = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,WATER_SATURATION).value\n")
    if SOIL:
        fileID.write("	cSOIL = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,OIL_SATURATION).value\n")
    fileID.write("\n")
    # fileID.write("	Data = [cID[0], cI, cJ, cK, cPress, cSWAT]\n")
    fileID.write("	Data = [cID[0], cI, cJ, cK, cPress")

    if SWAT:
        fileID.write(", cSWAT")
    if SOIL:
        fileID.write(",	cSOIL")
    fileID.write("]\n")

    fileID.write("	allData = list(zip(*Data))\n\n")

    fileID.write("	PRESSUREGrid = np.transpose(np.array(cPress).reshape((nZ, nY, nX)))\n")
    if SWAT:
        fileID.write("	SWATGrid = np.transpose(np.array(cSWAT).reshape((nZ, nY, nX)))\n")
    if SOIL:
        fileID.write("	SOILGrid = np.transpose(np.array(cSOIL).reshape((nZ, nY, nX)))\n")

    fileID.write("\n")

    fileID.write("	return allData, PRESSUREGrid")
    if SWAT:
        fileID.write(", SWATGrid")
    if SOIL:
        fileID.write(", SOILGrid")
    fileID.write("\n\n")

    # Dump into .pkl files:

    TimeVector = np.linspace(0, SimulationSettings.TotalLife, num=(SimulationSettings.NumberOfCycles+1))
    fileID.write("reportTimes = " + str(list(TimeVector)) + "\n")
    fileID.write("for rTime in reportTimes:\n")
    fileID.write("	rTimeIndex = reportTimes.index(rTime)\n")
    fileID.write("	if (rTime == FieldManagement().CurrentTime.get() ):\n")
    fileID.write("		print('Writing requested report at time: ', rTime, ' days...')\n")
    # fileID.write("		nData, PRESSUREGrid, SWATGrid = report(rTime)\n")

    fileID.write("		nData, PRESSUREGrid")
    if SWAT:
        fileID.write(", SWATGrid")
    if SOIL:
        fileID.write(", SOILGrid")
    fileID.write(" = report(rTime)\n")

    if naming_mode == 'time':
        fileID.write("		_filenamePress = _basefilename + \"_Time\" + str(rTime) + 'PRESSURE' + extension\n")
        if SWAT:
            fileID.write("		_filenameSWAT = _basefilename + \"_Time\" + str(rTime) + 'SWAT' + extension\n")
        if SOIL:
            fileID.write("		_filenameSOIL = _basefilename + \"_Time\" + str(rTime) + 'SOIL' + extension\n")
        fileID.write("\n")
    elif naming_mode == 'control-step':
        fileID.write("		_filenamePress = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'PRESSURE' + extension\n")
        if SWAT:
            fileID.write("		_filenameSWAT = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'SWAT' + extension\n")
        if SOIL:
            fileID.write("		_filenameSOIL = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'SOIL' + extension\n")
        fileID.write("\n")

    fileID.write("		with open(_filenamePress, 'wb') as _pklfile:\n")
    fileID.write("		    pickle.dump(PRESSUREGrid, _pklfile)\n\n")
    # fileID.write("		_pklfile.close()\n\n")

    if SWAT:
        fileID.write("		with open(_filenameSWAT, 'wb') as _pklfile:\n")
        fileID.write("		    pickle.dump(SWATGrid, _pklfile)\n\n")
    if SOIL:
        fileID.write("		with open(_filenameSOIL, 'wb') as _pklfile:\n")
        fileID.write("		    pickle.dump(SOILGrid, _pklfile)\n\n")

    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the final Script:
    fileID.write("	FinalScript = @{\n")
    fileID.write("}@\n\n")
    fileID.write("}\n\n")

    ##################################################################################################
    ############## NEWLY ADDED ###############################################
    ###########################################
    fileID.write("CustomControl \"GRID_PRESSURE_SWAT_INIT\" {\n")
    fileID.write("	ExecutionPosition=BEGIN_TIMESTEP")
    fileID.write("\n\n")

    #################################################################################################################
    # Write the InitialScript:
    fileID.write("	InitialScript = @{\n")
    fileID.write("import pickle" + '\n')
    fileID.write("import numpy as np" + '\n')
    fileID.write("_basefilename = Reservoir().get_name() " + "+ \"_Report\"" + '\n')
    fileID.write("extension = \".pkl\"" + '\n')
    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the main Script:
    fileID.write("	Script=@{\n")

    #Function Definition:
    fileID.write("def report(rTime):\n")
    fileID.write("	allData = []\n")
    fileID.write("	fCellId = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").FirstCellId.get()\n")
    fileID.write("	nX = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInI.get()\n")
    fileID.write("	nY = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInJ.get()\n")
    fileID.write("	nZ = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + "\").NumberCellsInK.get()\n")
    fileID.write("	cSelectionName = 'MyReservoir'\n")
    fileID.write("	selectionName = 'ResCells'\n")
    fileID.write("	cellsSelected = []\n")
    fileID.write("	Data = []\n")
    fileID.write("	cI = []\n")
    fileID.write("	cJ = []\n")
    fileID.write("	cK = []\n")
    fileID.write("	for k in range(nZ):\n")
    fileID.write("		for j in range(nY):\n")
    fileID.write("			for i in range(nX):\n")
    fileID.write("				cellId = fCellId + (i) + (j)*nX + (k)*nX*nY\n")
    fileID.write("				cellsSelected.append(cellId)\n")
    fileID.write("				cI.append(i + 1)\n")
    fileID.write("				cJ.append(j + 1)\n")
    fileID.write("				cK.append(k + 1)\n")
    fileID.write("	CellSelectionFamily(cSelectionName).add_selection(selectionName, cellsSelected)\n")
    fileID.write("	cID = GridMgr('GridMgr').RegionFamilyMgr('RegionFamilyMgr').CellSelectionFamily(cSelectionName).Cells.value\n")
    fileID.write("	cPress = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,PRESSURE).value\n")

    if SWAT:
        fileID.write("	cSWAT = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,WATER_SATURATION).value\n")
    if SOIL:
        fileID.write("	cSOIL = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,OIL_SATURATION).value\n")
    fileID.write("\n")
    # fileID.write("	Data = [cID[0], cI, cJ, cK, cPress, cSWAT]\n")
    fileID.write("	Data = [cID[0], cI, cJ, cK, cPress")

    if SWAT:
        fileID.write(", cSWAT")
    if SOIL:
        fileID.write(",	cSOIL")
    fileID.write("]\n")

    fileID.write("	allData = list(zip(*Data))\n\n")

    fileID.write("	PRESSUREGrid = np.transpose(np.array(cPress).reshape((nZ, nY, nX)))\n")
    if SWAT:
        fileID.write("	SWATGrid = np.transpose(np.array(cSWAT).reshape((nZ, nY, nX)))\n")
    if SOIL:
        fileID.write("	SOILGrid = np.transpose(np.array(cSOIL).reshape((nZ, nY, nX)))\n")

    fileID.write("\n")

    fileID.write("	return allData, PRESSUREGrid")
    if SWAT:
        fileID.write(", SWATGrid")
    if SOIL:
        fileID.write(", SOILGrid")
    fileID.write("\n\n")

    # Dump into .pkl files:

    fileID.write("reportTimes = " + str([0.0]) + "\n")
    fileID.write("for rTime in reportTimes:\n")
    fileID.write("	rTimeIndex = reportTimes.index(rTime)\n")
    fileID.write("	if (rTime == FieldManagement().CurrentTime.get() ):\n")
    fileID.write("		print('Writing requested report at time: ', rTime, ' days...')\n")
    # fileID.write("		nData, PRESSUREGrid, SWATGrid = report(rTime)\n")

    fileID.write("		nData, PRESSUREGrid")
    if SWAT:
        fileID.write(", SWATGrid")
    if SOIL:
        fileID.write(", SOILGrid")
    fileID.write(" = report(rTime)\n")

    if naming_mode == 'time':
        fileID.write("		_filenamePress = _basefilename + \"_Time\" + str(rTime) + 'PRESSURE' + extension\n")
        if SWAT:
            fileID.write("		_filenameSWAT = _basefilename + \"_Time\" + str(rTime) + 'SWAT' + extension\n")
        if SOIL:
            fileID.write("		_filenameSOIL = _basefilename + \"_Time\" + str(rTime) + 'SOIL' + extension\n")
        fileID.write("\n")
    elif naming_mode == 'control-step':
        fileID.write("		_filenamePress = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'PRESSURE' + extension\n")
        if SWAT:
            fileID.write("		_filenameSWAT = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'SWAT' + extension\n")
        if SOIL:
            fileID.write("		_filenameSOIL = _basefilename + \"_ControlStep\" + str(rTimeIndex) + 'SOIL' + extension\n")
        fileID.write("\n")

    fileID.write("		with open(_filenamePress, 'wb') as _pklfile:\n")
    fileID.write("		    pickle.dump(PRESSUREGrid, _pklfile)\n\n")
    # fileID.write("		_pklfile.close()\n\n")

    if SWAT:
        fileID.write("		with open(_filenameSWAT, 'wb') as _pklfile:\n")
        fileID.write("		    pickle.dump(SWATGrid, _pklfile)\n\n")
    if SOIL:
        fileID.write("		with open(_filenameSOIL, 'wb') as _pklfile:\n")
        fileID.write("		    pickle.dump(SOILGrid, _pklfile)\n\n")

    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the final Script:
    fileID.write("	FinalScript = @{\n")
    fileID.write("}@\n\n")
    fileID.write("}")

    fileID.close()


    return None


def CustomControlPS_CSV_PKL(CustomScriptsfileID, SimulationSettings, naming_mode='control-step', StructuredInfoGrid='CoarseGrid', SWAT=True, SOIL=False):
    fileID = open(CustomScriptsfileID, "w")
    fileID.write("MODEL_DEFINITION\n\n\n")

    fileID.write("################################################################################################################\n")
    fileID.write("# THIS SCRIPT EXTRACT PRESSURE AND WATER SATURATION INFORMATION OF THE GRIDBLOCKS AND EXPORT IT INTO .CSV FILES AND PICKLE  (.PKL) FILES \n \n")
    fileID.write("################################################################################################################\n")

    fileID.write("CustomControl \"GRID_PRESSURE_SWAT\" {\n")
    fileID.write("	ExecutionPosition=END_TIMESTEP")
    fileID.write("\n\n")

    #################################################################################################################
    # Write the InitialScript:
    fileID.write("	InitialScript = @{\n")
    fileID.write("import csv" + '\n')
    fileID.write("import pickle" + '\n')
    fileID.write("import numpy as np" + '\n')
    fileID.write("_basefilename = Reservoir().get_name() " + "+ \"_Report\"" + '\n')
    fileID.write("extension = \".csv\"" + '\n')
    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the main Script:
    fileID.write("	Script=@{\n")

    # Function Definition:
    fileID.write("def report(rTime, _filename):\n")
    fileID.write("	_file = open(_filename, \'w\', newline=\'\')\n")
    fileID.write("	allData = []\n")
    fileID.write("	writer = csv.writer(_file, delimiter=\',\', quotechar=\'|\', quoting=csv.QUOTE_MINIMAL)\n")
    fileID.write("	writer.writerow([\'Report at :\' + str(rTime) + \' days\'])\n")
    # fileID.write("	writer.writerow([\'CellID\', \'Cell_I\', \'Cell_J\', \'Cell_K\', \'Pressure (PRESSURE)\',\'Water Saturation (SWAT)\'])\n")
    fileID.write("	writer.writerow([\'CellID\', \'Cell_I\', \'Cell_J\', \'Cell_K\', \'Pressure (PRESSURE)\'")

    if SWAT:
        fileID.write(",\'Water Saturation (SWAT)\'")
    if SOIL:
        fileID.write(",\'Oil Saturation (SOIL)\'")

    fileID.write("])\n")

    fileID.write("	fCellId = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + ").FirstCellId.get()\n")
    fileID.write("	nX = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + ").NumberCellsInI.get()\n")
    fileID.write("	nY = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + ").NumberCellsInJ.get()\n")
    fileID.write("	nZ = GridMgr('GridMgr').StructuredInfoMgr('StructuredInfoMgr').StructuredInfo(\"" + StructuredInfoGrid + ").NumberCellsInK.get()\n")
    fileID.write("	cSelectionName = 'MyReservoir'\n")
    fileID.write("	selectionName = 'ResCells'\n")
    fileID.write("	cellsSelected = []\n")
    fileID.write("	Data = []\n")
    fileID.write("	cI = []\n")
    fileID.write("	cJ = []\n")
    fileID.write("	cK = []\n")
    fileID.write("	for k in range(nZ):\n")
    fileID.write("		for j in range(nY):\n")
    fileID.write("			for i in range(nX):\n")
    fileID.write("				cellId = fCellId + (i) + (j)*nX + (k)*nX*nY\n")
    fileID.write("				cellsSelected.append(cellId)\n")
    fileID.write("				cI.append(i + 1)\n")
    fileID.write("				cJ.append(j + 1)\n")
    fileID.write("				cK.append(k + 1)\n")
    fileID.write("	CellSelectionFamily(cSelectionName).add_selection(selectionName, cellsSelected)\n")
    fileID.write("	cID = GridMgr('GridMgr').RegionFamilyMgr('RegionFamilyMgr').CellSelectionFamily(cSelectionName).Cells.value\n")
    fileID.write("	cPress = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,PRESSURE).value\n")

    if SWAT:
        fileID.write("	cSWAT = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,WATER_SATURATION).value\n")
    if SOIL:
        fileID.write("	cSOIL = GridMgr('GridMgr').GridPropertyMgr('GridPropertyMgr').get_selection_property_values(cSelectionName,OIL_SATURATION).value\n")
    fileID.write("\n")
    # fileID.write("	Data = [cID[0], cI, cJ, cK, cPress, cSWAT]\n")
    fileID.write("	Data = [cID[0], cI, cJ, cK, cPress")

    if SWAT:
        fileID.write(", cSWAT")
    if SOIL:
        fileID.write(",	cSOIL")
    fileID.write("]\n")

    fileID.write("	allData = list(zip(*Data))\n\n")

    fileID.write("	for i in range(len(allData)):\n")
    # fileID.write("		writer.writerow([allData[i][0], allData[i][1], allData[i][2], allData[i][3], allData[i][4], allData[i][5]])\n")
    fileID.write("		writer.writerow([data for data in allData[i])\n")
    fileID.write("	_file.close()\n\n")

    # Dump into .pkl files:
    fileID.write("	PRESSUREGrid = np.transpose(np.array(cPress).reshape((nZ, nY, nX)))\n")
    fileID.write("	_pklPress = _filename.replace(\".csv\", \"PRESSURE.pkl\")\n")
    fileID.write("	with open(_pklPress, 'wb') as _pklfile:\n")
    fileID.write("	    pickle.dump(PRESSUREGrid, _pklfile)\n\n")
    # fileID.write("	_pklfile.close()\n\n")

    if SWAT:
        fileID.write("	SWATGrid = np.transpose(np.array(cSWAT).reshape((nZ, nY, nX)))\n")
        fileID.write("	_pklSWAT = _filename.replace(\".csv\", \"SWAT.pkl\")\n")
        fileID.write("	with open(_pklSWAT, 'wb') as _pklfile:\n")
        fileID.write("	    pickle.dump(SWATGrid, _pklfile)\n\n")
    if SOIL:
        fileID.write("	SOILGrid = np.transpose(np.array(cSOIL).reshape((nZ, nY, nX)))\n")
        fileID.write("	_pklSOIL = _filename.replace(\".csv\", \"SOIL.pkl\")\n")
        fileID.write("	with open(_pklSOIL, 'wb') as _pklfile:\n")
        fileID.write("	    pickle.dump(SOILGrid, _pklfile)\n\n")
    fileID.write("\n")

    fileID.write("	return allData\n\n")

    TimeVector = np.linspace(0, SimulationSettings.TotalLife, num=(SimulationSettings.NumberOfCycles + 1))
    fileID.write("reportTimes = " + str(list(TimeVector)) + "\n")
    fileID.write("for rTime in reportTimes:\n")
    fileID.write("	rTimeIndex = reportTimes.index(rTime)\n")
    fileID.write("	if (rTime == FieldManagement().CurrentTime.get() ):\n")
    if naming_mode == 'time':
        fileID.write("		_filename = _basefilename + \"_Time\" + str(rTime) + extension\n")
    elif naming_mode == 'control-step':
        fileID.write("		_filename = _basefilename + \"_ControlStep\" + str(rTimeIndex) + extension\n")
    fileID.write("		print('Writing requested report at time: ', rTime, ' days...')\n")
    fileID.write("		nData = report(rTime, _filename)\n")
    fileID.write("}@\n\n")

    #################################################################################################################
    # Write the final Script:
    fileID.write("	FinalScript = @{\n")
    fileID.write("}@\n\n")
    fileID.write("}")

    fileID.close()

    return None

def CustomControl_PRESS_SWAT(IncludeLinks, SimulationSettings, export_csv=False, name_suffix='control-step', StructuredInfoGrid='CoarseGrid', SWAT=True, SOIL=False):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    for Realization in range(NumberOfRealizations):
        if name_suffix.lower() != 'control-step' and name_suffix.lower() != 'time':
            print("Invalid name suffix input! Automatically setting this value to \'time\'...")
            name_suffix = 'time'

        if export_csv:
            CustomControlPS_CSV_PKL(IncludeLinks.CustomScriptForPRESSUREandSWAT[Realization], SimulationSettings, naming_mode=name_suffix.lower(),
                                    StructuredInfoGrid=StructuredInfoGrid, SWAT=SWAT, SOIL=SOIL)
        else:
            CustomControlPS_PKL(IncludeLinks.CustomScriptForPRESSUREandSWAT[Realization], SimulationSettings, naming_mode=name_suffix.lower(),
                                    StructuredInfoGrid=StructuredInfoGrid, SWAT=SWAT, SOIL=SOIL)

    return None
