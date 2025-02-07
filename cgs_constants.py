### this python file has a list of most used cgs units
### call this as -- import cgs_constants as cgs
### cgs.Msun -- for mass of sun (as an example) 

import numpy as np
import math

### stephan boltzmann constant
sb = 5.6705e-5
### Thompson scattering cross section
sigth= 6.6500e-25
### gravitaional constant 
G = 6.6726e-8
### boltzman's constant
kb = 1.38e-16
### luminosity of sun
Lsun = 3.839e33
### radius of sun
Rsun = 6.955e10
### mass of sun
Msun = 1.99e33
### speed of light
c= 2.99792458e10
### electron scattering opacity for solar metallicity
kappa_e = 0.34
### electron scattering opacity for WR-star metallicity (Y= 0.98, Z=0.02)
kappa_wr = 0.2
### mass of electron
mass_e= 9.1094e-28
### mass of proton/hydrogen
mass_p= 1.6726e-24
### mass of neutron
mass_n= 1.6749286e-24	 
### charge of electron
q_e= 4.8032e-10
### Radius of electron
radius_e= 2.8179e-13 
### Plank's constant
plank_h= 6.6261e-27
### Rydbergs' constant
ryd= 1.0974e5
### Bohr radius
bohr_r = 5.2918e-9
### Atomic cross section
atom_cs = 8.7974e-17
### Avogardo's number
avgd_num= 6.0221e23
### Gas constant
R_gas= 8.3145e7
### atomic mass unit
amu= 1.6605402e-24
### erg -- electron volt
eV = 6.2415e11
### seconds in a year
year= 31556926
### mass-loss per year in solar units
mdot_sun= year/Msun
### light year
ly= 9.461e17
### parsec
pc= 3.086e18
### Astronomical unit
Au= 1.496e13
### mass of earth
M_earth= 5.974e27
### mass of jupiter
M_jupiter= 1.899e30
### radius of earth
R_earth = 6.378e8
### radius of jupiter
R_jupiter= 7.149e9

if __name__ == "__main__":
    main()

