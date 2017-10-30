# Experiment 2: Comparing the Matrix based and MatrixFree method for vector valued function, without solver

## Introduction
This code will numerically solve the Laplace equation as defined in [step-20](https://www.dealii.org/8.5.0/doxygen/deal.II/step_20.html) with the strategies in [step-22](https://www.dealii.org/8.5.0/doxygen/deal.II/step_22.html)
It will further solve the same Laplace equation using [step-37](https://www.dealii.org/8.5.0/doxygen/deal.II/step_37.html) without the solver step(currently)

## FE
Q(k+1),Q(k) elements in 2D, with degrees 1

TBD: 2D, with degrees 2 and 3
TBD: 3D, with degrees 2 and 3

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

	bash -x make.sh 20
	bash -x make.sh 20-1
	bash -x make.sh 37-1
	
	#These inturn call
	#cmake . -DDEAL_II_DIR=/path/to/deal.II
	#make -j4
