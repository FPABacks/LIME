# This is an commentary line in a makefile
# Start of the makefile
ifndef MFORCE_DIR
$(error MFORCE_DIR in not set)
endif

src = $(MFORCE_DIR)/src


make_Run: Compile_f90getopt Compile_CGS_constants Compile_LTE_Line_module Compile_Run_modul
	gfortran -o run $(src)/f90getopt.o $(src)/LTE_Line_module.o $(src)/CGS_constants.o Run_module.o

Compile_Run_modul: $(src)/f90getopt.mod $(src)/cgs_constants.mod $(src)/lte_line_module.mod ./Run_module.f90
	gfortran -c Run_module.f90 -I $(src)/.

Compile_f90getopt: $(src)/f90getopt-master/f90getopt.F90
	gfortran -c $(src)/f90getopt-master/f90getopt.F90 -o $(src)/f90getopt.o -J $(src)/.

Compile_LTE_Line_module: $(src)/LTE_Line_module.f90 
	gfortran -c $(src)/LTE_Line_module.f90 -o $(src)/LTE_Line_module.o -J $(src)/.

Compile_CGS_constants: $(src)/CGS_constants.f90
	gfortran -c $(src)/CGS_constants.f90 -o $(src)/CGS_constants.o -J $(src)/.

clean:
	rm  *.o *.mod

allclean:
	rm $(src)/*.o $(src)/*.mod
	rm  run

# End of the makefile
