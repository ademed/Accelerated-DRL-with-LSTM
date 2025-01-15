import sys
import os

from E2CO.Misc.BlenderColorScript import BlenderColor
bcolors = BlenderColor()

class IncludeLinks:
    """
    Inputs:
        simulator: Simulator type - "IMEX" or "GEM"

    Class attributes:
        IncludeLinks.SimulatorDirectory: file directory where the simulator (IMEX or GEM) is located.
        IncludeLinks.BatchFile: file directory where the user wants the batch file (.bat) to be written.
        IncludeLinks.SimulatorType: "IMEX" or "GEM"
        IncludeLinks.ReportDirectory: file directory where the simulator (for example, IMEX - mx201611.exe or GEM - gm201611.exe) is located.
        IncludeLinks.InjectorResults: TWO (2) file directories the user wants the injector results to be written, first element is the .rwd file,
                          second element is the .rwo file.
        IncludeLinks.ProducersResults: TWO (2) file directories the user wants the producer results to be written, first element is the .rwd file,
                          second element is the .rwo file.
        IncludeLinks.AFIDataFile: file directory where the simulation datafile (.dat) is located.
        IncludeLinks.Schedule: file directory where the INCLUDE file (SCHEDULE.inc) is located, to link with the simulation datafile (.dat)
        IncludeLinks.Numerical: file directory where the INCLUDE file (NUMERICAL.inc) is located, to link with the simulation datafile (.dat)
        IncludeLinks.Wells: file directory where the INCLUDE file (WELLS.inc) is located, to link with the simulation datafile (.dat)
        IncludeLinks.NumberOfRealizations: total number of geological realizations.
        IncludeLinks.MainFolder: file directory at which the folder that contains all the geological realizations (ModelFolderName) is located.
        IncludeLinks.ModelFolderName: name of the parent folder that contains all the realizations.
        IncludeLinks.SubfolderName: prefix of the child folder name for each realization within the parent folder ModelFolderName.

    """

    def __init__(self, version="2020.4", NumberOfRealizations=10, 
                 MainFolder =r'C:\Reinforcement-Learning-Codes--zero-to-hero-\Using RayLib on custom gym environment\IX_DeterministicOptimization', 
                 ModelFolderName='ChannelModel', SubfolderName='Realization',
                 AFIDataFile = 'BASE.afi', ScheduleFile = 'Schedule.ixf'):
        self.Version = version
        self.SimulatorDirectory = r"C:\ecl\macros\eclrun.exe"
        self.BatchFile = []
        for i in range(NumberOfRealizations):
            batchfile = os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i+1) + '\\' + 'batch_file_ix.bat')
            self.BatchFile.append(batchfile)

        self.AFIDataFile = []
        self.Schedule = []
        self.CustomScripts = []
        self.CustomScriptsForNPV = []
        self.CustomScriptForPRESSUREandSWAT = []
        self.PRTFile = []
        self.MSGFile = []
        for i in range(NumberOfRealizations):
            self.AFIDataFile.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + AFIDataFile))
            self.Schedule.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + ScheduleFile))
            self.CustomScripts.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + 'CustomScripts.ixf'))
            self.CustomScriptsForNPV.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + 'CustomScriptForNPV.ixf'))
            self.CustomScriptForPRESSUREandSWAT.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + 'CustomScriptGRID_PRESSURE_SWAT.ixf'))
            self.PRTFile.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + AFIDataFile + '.PRT' ))
            self.MSGFile.append(os.path.join(MainFolder + '\\' + ModelFolderName + '\\' + SubfolderName + str(i + 1) + '\\' + AFIDataFile + '.MSG'))
            #self.PRTFile.append(os.path.join(os.path.split(self.AFIDataFile[i])[0], os.path.split(self.AFIDataFile[i])[1].replace(".afi", ".PRT")))
            #self.MSGFile.append(os.path.join(os.path.split(self.AFIDataFile[i])[0], os.path.split(self.AFIDataFile[i])[1].replace(".afi", ".MSG")))



class Bounds:
    """
    Inputs:
        LowerBHP: Lower bound for BHP (default = 500)
        UpperBHP: Upper bound for BHP (default = 2000)
        LowerWinj: Lower bound for water injecton rate (default = 0)
        UpperWinj: Upper bound for water injecton rate (default = 10000)

    Class attributes:
        WellNames.Injectors: a list consisting of all the injector well names created from the prefixes (e.g. BR-I-1, BR-I-2,...)
        WellNames.Producers: a list consisting of all the producer well names created from the prefixes (e.g. BR-P-1, BR-P-2,...)
    """
    def __init__(self, LowerBHP=500, UpperBHP=1000, LowerWinj=5000, UpperWinj=10000):
        self.LowerBHP = LowerBHP
        self.UpperBHP = UpperBHP
        self.LowerWinj = LowerWinj
        self.UpperWinj = UpperWinj


class WellNamesDefinition:
    """
    Inputs:
        Ninj: Number of injectors
        Nprd: Number of producers
        InjPrefix: Injector name prefix (default = "I")
        PrdPrefix: Producer name prefix (default = "P")
        GroupName: name of the group that contains all the wells

    Class attributes:
        WellNames.Injectors: a list consisting of all the injector well names created from the prefixes (e.g. I1, I2,...)
        WellNames.Producers: a list consisting of all the producer well names created from the prefixes (e.g. P1, P2,...)
    """

    def __init__(self, Ninj=10, Nprd=20, InjPrefix="I", PrdPrefix="P", GroupName='G1'):
        vec1 = []
        vec2 = []
        for i in range(Ninj):
            vec1.append(InjPrefix + str(i+1))
        for i in range(Nprd):
            vec2.append(PrdPrefix + str(i+1))

        self.Injectors = vec1
        self.Producers = vec2
        self.GroupName = GroupName
