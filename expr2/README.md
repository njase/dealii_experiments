# Experiment 2: Comparing the Matrix based and MatrixFree method for vector valued function

## Introduction
Step-20.cc - Only added extra logs to original step-20 (Matrix based vector valued solver - Mixed laplace eqn)
Step20-1.cc - Step-20 + replaced RT with FE_Q + changed degree from 0 to 2, changed dim from 2 to 3 (This is like in step-37)
Step20-2.cc - Step20-1 + modified using MatrixFree (TBD: solver??)

## FE
Q(k+1),Q(k) elements

## Remarks
2. No MPI

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

	bash -x make.sh 20
	bash -x make.sh 20-1
	bash -x make.sh 20-2
	
	#These inturn call
	#cmake . -DDEAL_II_DIR=/path/to/deal.II
	#make -j4
