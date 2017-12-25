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

cmake -DDEAL_II_WITH_LAPACK=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=. ../../dealii/
make -j4
