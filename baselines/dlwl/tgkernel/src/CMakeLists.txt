# This file is part of TGKernel.
# Copyright (C) 2020 Lutz Oettershagen
# Contact: lutz.oettershagen@cs.uni-bonn.de


project(tgkernel)

# set source files
set(SOURCES
        tgkernel_main.cpp
        TemporalGraphs.cpp
        DataSetLoader.cpp
        NonTemporalGraphKernel.cpp
        StaticExpansionKernel.cpp
        GraphKernelCommonFunctions.cpp
        Timer.cpp
        StaticGraphKernel.cpp
        TGRepresentations.cpp
        TemporalGraphKernelApprox.cpp
        ExplicitRW.cpp
        ExplicitLineGraphKernel.cpp
        TGStatistics.cpp

        # WL Kernel files
        ../lib/wlkernel/AuxiliaryMethods.cpp
        ../lib/wlkernel/ColorRefinementKernel.cpp
        ../lib/wlkernel/Graph.cpp
        )

# define target
add_executable(tgkernel ${SOURCES})

set(EXECUTABLE_OUTPUT_PATH "..")
