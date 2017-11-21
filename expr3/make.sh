#!/bin/bash

#TBD: Add error checks


export CXX=/export/home/smehta/opt/clang5.0.0/bin/clang++
export CC=/export/home/smehta/opt/clang5.0.0/bin/clang
rm -fr CMakeCache.txt CMakeFiles
cmake . -DDEAL_II_DIR=../../install_dealii/clang_build
make -j4
