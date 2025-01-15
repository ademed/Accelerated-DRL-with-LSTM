########################################################################################################################
# Author: Quang M. Nguyen (qun972@utulsa.edu)
# This .py script contains the Blender codes for different colors, used in changing text colors within print() commands
# Helps with neater/better visualization on Python IDE Command Windows
# To add more colors, simply modify the Python class definition below
#
########################################################################################################################
class BlenderColor:
    def __init__(self):
        self.HEADER = '\033[95m'
        self.OKBLUE = '\033[94m'
        self.OKCYAN = '\033[96m'
        self.OKGREEN = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'

########################################################################################################################