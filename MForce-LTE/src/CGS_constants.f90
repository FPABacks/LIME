MODULE CGS_constants
  REAL(8), PARAMETER :: sigma_stef = 5.670367000d-05
  REAL(8), PARAMETER :: sigma_clas = 0.026540000d+00
  REAL(8), PARAMETER :: sigma_Thom = 6.652458732d-25
  REAL(8), PARAMETER :: radi_const = 7.565700000d-15
  REAL(8), PARAMETER :: plnk_const = 6.626070150d-27

  REAL(8), PARAMETER :: bolz_const = 1.380648520d-16
  REAL(8), PARAMETER :: Rgas_const = 8.314459800d+07
  REAL(8), PARAMETER :: elect_mass = 9.109383560d-28
  REAL(8), PARAMETER :: proto_mass = 1.6726d-24
  REAL(8), PARAMETER :: hydro_mass = 1.673723600d-24
  REAL(8), PARAMETER :: helli_mass = 6.646476400d-24
  REAL(8), PARAMETER :: N_avogadro = 6.022140857d+23

  REAL(8), PARAMETER :: lght_speed = 2.99792458d+10
  REAL(8), PARAMETER :: grav_const = 6.67191000d-08

  REAL(8), PARAMETER :: solar_mass = 1.98855d+33
  REAL(8), PARAMETER :: solar_radi = 6.95700d+10
  REAL(8), PARAMETER :: solar_lumi = 3.82800d+33
  REAL(8), PARAMETER :: solar_temp = 5.78000d+03

  REAL(8), PARAMETER :: Aa2cgs  = 1.000d-08
  REAL(8), PARAMETER :: cm2cgs  = 1.000d+00
  REAL(8), PARAMETER :: m2cgs   = 1.000d+02
  REAL(8), PARAMETER :: km2cgs  = 1.000d+05
  REAL(8), PARAMETER :: au2base  = 1.496d+13
  REAL(8), PARAMETER :: pc2cgs  = 3.086d+18
  REAL(8), PARAMETER :: ly2cgs  = 9.463d+17

  REAL(8), PARAMETER :: sec2cgs = 1.000d+00
  REAL(8), PARAMETER :: min2cgs  = 6.000d+01
  REAL(8), PARAMETER :: hour2cgs = 3.600d+03
  REAL(8), PARAMETER :: year2cgs = 3.15576d+07

  REAL(8), PARAMETER :: gr2cgs = 1.0d+00
  REAL(8), PARAMETER :: kg2cgs = 1.0d-03

  REAL(8), PARAMETER :: eV2cgs  = 1.60218d-12
  REAL(8), PARAMETER :: jul2cgs = 1.0d+07

  REAL(8), PARAMETER :: pi = 3.141592653589793d+00

END MODULE CGS_constants
