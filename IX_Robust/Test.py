import numpy as np
import pandas as pd

from CMG_Robust.ReadInput import IncludeLinks, Bounds, WellNamesDefinition
from CMG_Robust.Writers import ConversionFactors, SimulationSettings, NumericalSettings, RwdSettings, NumericalWriter, ScheduleWriter, RwdGroupWriter
from CMG_Robust.RunCMG import BatchFileWriter, RunCMG, RunCMGAll
from CMG_Robust.Readers import RwoGroupReader, RwoGroupReaderAll
from SQPOptimizer_Robust.Functions import *
IncludeLinks = IncludeLinks(simulator="IMEX", NumberOfRealizations=5)

SimulationSettings = SimulationSettings()
well_names = WellNamesDefinition(Ninj=10, Nprd=20)
Bounds = Bounds()
ConversionFactors = ConversionFactors()
NumericalSettings = NumericalSettings()
RwdSettings = RwdSettings()

EconomicSettings = EconomicSettings()

NumberOfRealizations = 5
class well_controls:
    def __init__(self):
        self.Q = 10000*np.random.rand(10*SimulationSettings.NumberOfCycles, 1)
        self.BHP = 2000*np.random.rand(20*SimulationSettings.NumberOfCycles, 1)

well_controls = well_controls()

# Test batch file writer for ALL realizations:
BatchFileWriter(IncludeLinks)

# Test numerical settings for ALL realizations:
NumericalWriter(IncludeLinks, NumericalSettings, print_notice=True)

# Test schedule write function for ALL realizations:
ScheduleWriter(IncludeLinks, well_names, well_controls, ConversionFactors, SimulationSettings, print_notice=True)

# Test .rwd write function(s) for ALL realizations:
RwdGroupWriter(IncludeLinks, RwdSettings, print_notice=True)

# # Test CMG Autorun for ALL realizations:
RunCMGAll(IncludeLinks, print_simulation_cmd=False, time_simulation=True)

# # Test CMG Autorun a SINGLE realizations:
RunCMG(0, IncludeLinks, print_simulation_cmd=False, time_simulation=True)

# Test .rwo reader for a SINGLE realization:
[PrdData, InjData] = RwoGroupReader(1, IncludeLinks, RwdSettings)

# Test .rwo reader for ALL realizations:
[PrdDataAll, InjDataAll] = RwoGroupReaderAll(IncludeLinks, RwdSettings)

# Test NPV calcs:
[PrdDataFiltered, InjDataFiltered, CombinedDF, NPV1, TimeVector] = NPVCalc(PrdData, InjData, SimulationSettings, RwdSettings, EconomicSettings)
[NPV_avg, NPVs, CDFs] = NPVAverageCalc(PrdDataAll, InjDataAll, SimulationSettings, RwdSettings, EconomicSettings)

print(PrdData.shape)
print(InjData.shape)

a = np.array([1,2,3]).reshape((3,1))
b = np.array([4,5,6]).reshape((3,1))
c = np.array([7,8,9]).reshape((3,1))
d = np.array([10,11,12]).reshape((3,1))
dkm = [a, b, c, d]

vcl = np.hstack(dkm)
clgt = np.mean(vcl, axis=1, keepdims=True)