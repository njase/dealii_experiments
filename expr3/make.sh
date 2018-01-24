#!/bin/bash

#TBD: Add error checks


export CXX=/export/home/smehta/opt/clang5.0.0/bin/clang++
export CC=/export/home/smehta/opt/clang5.0.0/bin/clang
rm -fr CMakeCache.txt CMakeFiles
cmake . -DCMAKE_CXX_COMPILER=/home/smehta/opt/clang5.0.0/bin/clang++ -DDEAL_II_DIR=../../install/dealii_clang
make -j4
