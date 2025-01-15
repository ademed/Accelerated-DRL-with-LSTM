import sys
import os
import numpy as np
import pandas as pd
import locale

# from CMG.ReadInput import IncludeLinks, Bounds, WellNamesDefinition
# from CMG.Writers import ConversionFactors, SimulationSettings, NumericalSettings, RwdSettings, NumericalWriter, ScheduleWriter, RwdGroupWriter
# from CMG.RunCMG import BatchFileWriter, RunCMG
# from CMG.Readers import RwoGroupReader

from SQPOptimizer.Functions import DimensionCheck, Normalizer, Denormalizer

from Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()


def INSIMInitialization(well_names, SimulationSettings, Bounds, WWIRfile=r"C:\Users\qun972\PycharmProjects\SQP\Ying_CMG\initial\WWIRInitial.dat", WBHPfile=r"C:\Users\qun972\PycharmProjects\SQP\Ying_CMG\initial\WBHPInitial.dat"):
    Ninj = len(well_names.Injectors)
    Nprd = len(well_names.Producers)
    n_cyc = SimulationSettings.NumberOfCycles

    df2 = pd.read_csv(WBHPfile, header=None, sep=r'\t', engine='python')
    df1 = pd.read_csv(WWIRfile, header=None, sep=r'\t', engine='python')

    assert(df1.shape[0] == n_cyc)
    assert (df1.shape[1] == Ninj)
    assert(df2.shape[0] == n_cyc)
    assert (df2.shape[1] == Nprd)

    rates = []
    BHPs = []

    for i in range(Ninj):
        rates.append(df1.iloc[:, i])

    for j in range(Nprd):
        BHPs.append(df2.iloc[:, j])

    rates = np.array(rates).reshape(Ninj*n_cyc, 1)
    BHPs = np.array(BHPs).reshape(Nprd*n_cyc, 1)

    #
    # if not Randomized:
    #     rates = InitialValues[0]*np.ones((Ninj*n_cyc, 1))
    #     BHPs = InitialValues[1]*np.ones((Nprd*n_cyc, 1))
    # else:
    #     rates = np.random.rand(Ninj*n_cyc, 1)
    #     BHPs = np.random.rand(Nprd*n_cyc, 1)

    [rates, BHPs] = DimensionCheck(rates, BHPs)
    u0 = np.concatenate((rates, BHPs), axis=0)
    u0_norm = Normalizer(u0, well_names, SimulationSettings, Bounds)


    return [u0, u0_norm]