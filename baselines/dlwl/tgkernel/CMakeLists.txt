# This file is part of TGKernel.
# Copyright (C) 2020 Lutz Oettershagen
# Contact: lutz.oettershagen@cs.uni-bonn.de

cmake_minimum_required(VERSION 3.5)

project(TGKernel)

set(CMAKE_CXX_STANDARD 17)


# add include directories
include_directories(
        lib/

        # set the environment variable EIGEN3_INCLUDE_DIR to the
        # include directory of your Eigen3 installation
        $ENV{EIGEN3_INCLUDE_DIR}

        /home/lutz/Downloads/eigen-3.3.7

)

add_subdirectory(src)
