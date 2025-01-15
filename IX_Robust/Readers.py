import sys
import os

import numpy as np
import pandas as pd

from E2CO.Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()


# Function 1: Python Generator to read a file in REVERSED order:
def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment

# Function 1: Import .rwo files as tables/dataframes
def RwoGroupReader(Realization, IncludeLinks, RwdSettings):
    filepath = IncludeLinks.AFIDataFile[Realization].replace(".afi", "") + "_NPVData.csv"
    data = pd.read_csv(filepath, header=0)
    data.columns = data.columns.str.strip()
    PrdData = data.loc[:, ['TIME'] + RwdSettings.ProducersParameters]
    InjData = data.loc[:, ['TIME'] + RwdSettings.InjectorsParameters]

    if PrdData.shape[0] != InjData.shape[0]:
        sys.exit(f"{bcolors.BOLD}{bcolors.FAIL}FATAL ERROR! UNMATCHED IMPORTED DATAFRAME SIZES (FROM .RWO FILES)\n{bcolors.ENDC}")

    while PrdData.shape[1] > (len(RwdSettings.ProducersParameters) + 1):
        col = PrdData.shape[1] - 1
        PrdData = PrdData.drop(PrdData.columns[col], axis=1)

    while InjData.shape[1] > (len(RwdSettings.InjectorsParameters) + 1):
        col = InjData.shape[1] - 1
        InjData = InjData.drop(InjData.columns[col], axis=1)

    # if IncludeLinks.SimulatorType == "GEM":
    #     if "Cumulative Gas Mass(CO2) SC" in RwdSettings.ProducersParameters:
    #         PrdData = PrdData.rename(columns={"Cumulative Gas Mass$C SC": "Cumulative Gas Mass(CO2) SC"})
    #     if "Cumulative Gas Mass(CO2) SC" in RwdSettings.InjectorsParameters:
    #         InjData = InjData.rename(columns={"Cumulative Gas Mass$C SC": "Cumulative Gas Mass(CO2) SC"})

    return [PrdData, InjData]


# Function 2: Import .rwo files for each individual injector as tables/dataframes
def RwoGroupReaderAll(IncludeLinks, RwdSettings):
    NumberOfRealizations = len(IncludeLinks.BatchFile)
    PrdDataAll = []
    InjDataAll = []
    for Realization in range(NumberOfRealizations):
        [PrdData, InjData] = RwoGroupReader(Realization, IncludeLinks, RwdSettings)
        PrdDataAll.append(PrdData)
        InjDataAll.append(InjData)

    return [PrdDataAll, InjDataAll]

# Function 2: Import .rwo files for each individual injector as tables/dataframes
def RwoNLReader(filepath):
    data = pd.read_csv(filepath, header=0)
    data.columns = data.columns.str.strip()

    return data
