# Modified by Franziska Heeg, Jonas Sauer, Petra Mutzel, Ingo Scholtes (2026) for
# "Weisfeiler and Leman Follow the Arrow of Time" (NeurIPS 2026).
# Based on the TGKernel reference implementation: https://gitlab.com/tgpublic/tgkernel.
# Changes to the original file:
#   - C++17 instead of C++11 (Eigen 3.4)
# See PATCHES.md for details.

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
