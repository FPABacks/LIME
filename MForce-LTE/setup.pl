#!/usr/bin/perl -i

use strict;
use warnings;
use Getopt::Long;
use File::Copy;
use Env;
use Cwd;

my $help_message =
"Usage: setup.pl [options]      To set up an MFORCE-TLE computation

Options:

    -t=type                        type = 1  to give a CAK parameter problem 
                                   type = 2  to give a number density problem

Examples:

setup.pl -t=2\n";

# Define the option variables
my $type;
my $help;
my $dir = getcwd;

# Parse input and error checking.
GetOptions(
    "t=i"     =>
    sub {
        my ($opt_name, $opt_value) = @_;
        $opt_value =~ /([123])/ || die "-$opt_name Incorrect flag\n";
        $type = $1;
        $type >= 1 && $type <= 2 ||
            die("-t can be only 1 or 2\nPlease consult help\n");
    },
    "help"    => \$help)
    or die("Error in line arguments\n");

# Show help if -help is given or if there are no other arguments
if ($help || ! $type) {
    print STDERR $help_message;
    print STDERR "$dir\n";
    exit;
}

# Check if the environment variable AMRVAC_DIR is defined
if (!$ENV{MFORCE_DIR}) {
    print STDERR "Error: \$MFORCE_DIR not yet set, use:\n";
    print STDERR "export \$MFORCE_DIR=/destination\n";
    exit;
}

if ($type == 1) {
    copy("$ENV{MFORCE_DIR}/Run_module(CAK).f90","./Run_module.f90") or die "Copy ** Run_module(CAK).f90 ** failed: $!";
    copy("$ENV{MFORCE_DIR}/makefile_glob","./makefile") or die "Copy makefile failed: $!";
    copy("$ENV{MFORCE_DIR}/py_src/fit_v1.5.py","./fit_v1.5.py") or die "Copy ** fit_v1.5.py ** failed: $!";
    copy("$ENV{MFORCE_DIR}/CAK_example_Sun.par","./input_run.par") or die "Copy ** CAK_example_Sun.par ** failed: $!";
    copy("$ENV{MFORCE_DIR}/fit_example_sun.par","./input_fit.par") or die "Copy ** fit_example_sun.par ** failed: $!";
}elsif($type == 2){
    copy("$ENV{MFORCE_DIR}/Run_module(Line).f90","$dir/Run_module.f90") or die "Copy ** Run_module(Line).f90 ** failed: $!";
    copy("$ENV{MFORCE_DIR}/makefile_glob","$dir/makefile") or die "Copy ** makefile ** failed: $!";
    copy("$ENV{MFORCE_DIR}/Line_example_Sun.par","$dir/input_run.par") or die "Copy ** Line_example_Sun.par ** failed: $!";
}
