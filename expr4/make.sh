case $1 in
        1)
		cp -f temp/CMakeLists.txt_test1 CMakeLists.txt
                ;;
        2)
		cp -f temp/CMakeLists.txt_test2 CMakeLists.txt
                ;;
        *)
		cp -f temp/CMakeLists.txt_test CMakeLists.txt
                ;;
esac


export CXX="/home/smehta/opt/clang5.0.0/bin/clang++"
export CC="ccache /home/smehta/opt/clang5.0.0/bin/clang"
rm -fr CMakeCache.txt CMakeFiles
#SAURABH - noticed error on Uni machine that running cmake on examples fails to detect CXX compiler
#I guess the reason is that I used ccache for compiling dealii, and then the cmake cached variables from
#dealii are automatically take by the test program cmake lists. cmake has possibly some issue with
#reading CMAKE_CXX_COMPILER_ARG1 in automatic mode.
#Workaround - dont use ccache with test program, give CXX_COMPILER explicity in command to cmake
cmake -DCMAKE_CXX_COMPILER=/home/smehta/opt/clang5.0.0/bin/clang++ -DDEAL_II_DIR=../../install/dealii_clang .
make -j4
