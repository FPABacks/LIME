#!/bin/env bash

CodeDir="${MFORCE_DIR:-$PWD/MFORCE-LTE/}"
# There is a bug in f2py that causes it not to be able to read files in different locations.
# This is the most low-tech but fairly fool proof solution for now. Just creating a symbolic link for the
# compilation then get rid of those again.
ln -s $CodeDir/src/CGS_constants.f90 .
ln -s $CodeDir/src/LTE_Line_module.f90 .
python3 -m numpy.f2py -c CGS_constants.f90 LTE_Line_module.f90 Run_module.f90 only: get_force_multiplier -m mforce
rm CGS_constants.f90
rm LTE_Line_module.f90