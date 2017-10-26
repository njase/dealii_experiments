# Experiment 1: Comparing the Matrix based and MatrixFree method, without solver

## Introduction
This code will numerically solve the Poisson equation as defined in [step-37](https://www.dealii.org/8.5.0/doxygen/deal.II/step_37.html) with the strtegies in [step-5](https://www.dealii.org/8.5.0/doxygen/deal.II/step_5.html) and [step-37](https://www.dealii.org/8.5.0/doxygen/deal.II/step_37.html) without the solver step

## FE
FE_Q elements in 2D and 3D, with degrees 1, 2 and 3

## Remarks
1. No solver used
2. MPI support is removed from step-37 program

## Results to compare
1. Measurement of computation time for different phases with problem size = TBD DoF
2. Accuracy of results - (X) Not until solver is implemented
3. Overall solution time with varying problem size (DoF), only for MatrixFree method
4. Ovrall solution time while for a fixed problem size while varying the number of cores - (X) not possible on my machine
5. Imact of adaptivity - (X) Not at this point of time
6. Memory requirement - TBD
7. Cache performance
8. Others??



## Compiling and Running
To generate a makefile for this code using CMake and then compile in debug mode, type the following command 
into the terminal from the main directory:

	bash -x make.sh 5
	bash -x make.sh 37
	
	#These inturn call
	#cmake . -DDEAL_II_DIR=/path/to/deal.II
	#make -j4
