#!/bin/bash

#TBD: Add error checks

case $1 in
	20)
		export STEPVAR=20
		;;
	20-1)
		export STEPVAR=20-1
		;;
	20-2)
		export STEPVAR=20-2
		;;
	*)
		echo "Error in input ($1), should be either 20 or 20-1 or 20-2"
		exit
esac

rm -fr CMakeCache.txt CMakeFiles
cmake . -DDEAL_II_DIR=../../install/dealii_clang
make -j4
