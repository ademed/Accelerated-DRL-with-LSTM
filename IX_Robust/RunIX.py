import subprocess
import os
import numpy as np

from E2CO.Misc.BlenderColorScript import BlenderColor
from IX_Robust.Readers import reverse_readline
bcolors = BlenderColor()


## Function 1: Write the BATCH (.bat) file with the corresponding IX simulator
def BatchFileWriter(IncludeLinks, n_CPU=12, results_export=True, print_notice=False):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    for i in range(NumberOfRealizations):
        ####################################################################################################################
        fileID = open(IncludeLinks.BatchFile[i], "w+")
        # Begin of the .bat file:
        fileID.write("@echo off\n")
        fileID.write("SET LSHORST=no-net\n")
        fileID.write("@echo off\n\n")
        ####################################################################################################################
        # Main commands:
        fileID.write("\"" + IncludeLinks.SimulatorDirectory + "\"" + " -dd" + " -v " + IncludeLinks.Version + " --np " + str(n_CPU) + " ix" + " %1\n\n")

        ####################################################################################################################
        # End of the .bat file:
        fileID.write("CLS\nEXIT")
        fileID.close()

    ####################################################################################################################
    # Print out completion notice if requested:
    if print_notice:
        print(
            f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.OKGREEN}{bcolors.BOLD}Simulator's BATCH file (.bat) for INTERSECT v.-" + IncludeLinks.Version + f" is generated!{bcolors.ENDC}")

    return None


## Function 2: Run the CMG simulator
def RunIX(Realization, IncludeLinks, print_simulation_cmd=False, print_notice=True, time_simulation=False, check=True):
    x = not print_simulation_cmd
    import time

    null_results = 0
    t0 = time.time()
    while null_results == 0:
        subprocess.run([IncludeLinks.BatchFile[Realization], IncludeLinks.AFIDataFile[Realization]], capture_output=x)
        if check:
            null_results = CheckForNullResults(Realization, IncludeLinks, checkPRT=True, checkMSG=False)
        else:
            null_results = 1

    if null_results == 1 and x:
        if print_notice:
            print(f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.OKGREEN}{bcolors.BOLD}Simulation completed for realization #" + str(Realization+1) + f"!{bcolors.ENDC}")
    t1 = time.time()

    if time_simulation:
        print(f'{bcolors.BOLD}Time elapsed: ' + '{:.3f} seconds'.format(np.abs(t1 - t0)) + f'.{bcolors.ENDC}')
    return None


## Function 3: Check for any null result files from .PRT and .MSG files:
def CheckForNullResults(Realization, IncludeLinks, checkPRT=True, checkMSG=True):
    if checkPRT:
        readPRT = reverse_readline(IncludeLinks.PRTFile[Realization])
        linePRT = next(readPRT).replace('INFO', '').replace(' ', '').replace('.', '').lower()
        if linePRT == 'runfinishednormally':
            x1 = 1
        else:
            x1 = 0
    else:
        x1 = 1

    if checkMSG:
        readMSG = reverse_readline(IncludeLinks.MSGFile[Realization])
        for i in range(3):
            lineMSG = next(readMSG).replace('INFO', '').replace(' ', '').replace('.', '').lower()
            if lineMSG == 'runfinishednormally':
                x2 = 1
            else:
                x2 = 0
    else:
        x2 = 1

    if x1 * x2 != 0:
        out = 1
    else:
        out = 0

    return out


## Function 4: Run the CMG simulator for ALL realizations
def RunIXAll(IncludeLinks, print_simulation_cmd=False, print_notice=True, time_simulation=False, check=True):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    x = not print_simulation_cmd
    import time
    for i in range(NumberOfRealizations):
        if print_notice:
            print(f"{bcolors.OKBLUE}{bcolors.BOLD}Running simulation for Realization #" + str(i+1) + f"...{bcolors.ENDC}")
        null_results = 0
        if i > 0:
            del t0, t1
        t0 = time.time()
        while null_results == 0:
            subprocess.run([IncludeLinks.BatchFile[i], IncludeLinks.AFIDataFile[i]], capture_output=x)
            if check:
                null_results = CheckForNullResults(i, IncludeLinks, checkPRT=True, checkMSG=False)
            else:
                null_results = 1

        if null_results == 1 and x:
            if print_notice:
                print(f"{bcolors.WARNING}{bcolors.BOLD}Completion Notice: {bcolors.ENDC}" + f"{bcolors.OKGREEN}{bcolors.BOLD}Simulation completed!{bcolors.ENDC}")
        t1 = time.time()

        if time_simulation:
            print(f'{bcolors.BOLD}Time elapsed: ' + '{:.3f} seconds'.format(np.abs(t1 - t0)) + f'.{bcolors.ENDC}')

    return None