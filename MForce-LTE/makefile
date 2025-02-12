# This is an commentary line in a makefile
# Start of the makefile
ifndef MFORCE_DIR
$(info MFORCE_DIR Not found)
$(error Please execute: export MFORCE_DIR=$(pwd))
endif

make_Run: Compile_f90getopt Compile_CGS_constants Compile_LTE_Line_module Compile_Run_modul
	gfortran -o run f90getopt.o LTE_Line_module.o CGS_constants.o Run_module.o

Compile_Run_modul: f90getopt.mod cgs_constants.mod lte_line_module.mod Run_module.f90
	gfortran -c Run_module.f90

Compile_f90getopt: ./src/f90getopt-master/f90getopt.F90
	gfortran -c ./src/f90getopt-master/f90getopt.F90

Compile_LTE_Line_module: ./src/LTE_Line_module.f90
	gfortran -c ./src/LTE_Line_module.f90

Compile_CGS_constants: ./src/CGS_constants.f90
	gfortran -c ./src/CGS_constants.f90

clean:
	rm ./*.o ./*.mod
	rm  *.o *.mod
allclean: clean
	rm  run
# End of the makefile
