/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_TGSTATISTICS_H
#define TGKERNEL_TGSTATISTICS_H

#include "TemporalGraphs.h"

class TGStatistics {

public:

    void getTemporalGraphStreamsStatistics(TemporalGraphStreams &tgss);


private:

    void getData();

};


#endif //TGKERNEL_TGSTATISTICS_H
