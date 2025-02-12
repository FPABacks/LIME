import Fortran_namelist as fnml
import numpy as np
import os
import sys

GridSetup_File = 'test.par'
GridSetup_Lsn = 'setupa'
GridSetup_Nml = {
    'pq' : 0,
    'qp' : 0}
GridSetup_Mask = ['s','f']


try:
    GridSetup_Nml = fnml.Read_namelist(GridSetup_File,GridSetup_Lsn,GridSetup_Nml,GridSetup_Mask)
except Exception as Exc:
    print(Exc)
    quit()
else:
    print('Accessing LTE input parameter file ./' +  GridSetup_Nml['pq'])

print(GridSetup_Nml['qp'])