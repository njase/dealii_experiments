#!/bin/bash

#TBD: Add error checks

case $1 in
	5)
		export STEPVAR=5
		;;
	37)
		export STEPVAR=37
		;;
	*)
		echo "Error in input ($1), should be either 5 or 37"
		exit
esac

rm -fr CMakeCache.txt CMakeFiles
cmake . -DDEAL_II_DIR=../../install/dealii_clang
make -j4
