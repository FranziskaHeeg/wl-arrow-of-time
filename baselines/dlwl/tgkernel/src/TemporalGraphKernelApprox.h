/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_TEMPORALGRAPHKERNELAPPROX_H
#define TGKERNEL_TEMPORALGRAPHKERNELAPPROX_H

#include "TemporalGraphs.h"



class TemporalGraphKernelApprox {

public:

    void calculaleGramMatrix (TemporalGraphStreams &data, std::string datasetname, unsigned int kmin, unsigned int kmax, unsigned int S_size);

private:

    void getGramMatrix(TemporalGraphStreams &data, std::string datasetname, unsigned int k, unsigned int S_size);

    std::string seqToString(std::vector<Label> seq);

    std::vector<Label> temporalRandomWalkVertexLabel(TemporalGraphStream &tgs, NodeId s, unsigned int k);

    void getS(TemporalGraphStream &tgs, unsigned int S_size, std::vector<NodeId> &S);

    std::vector<std::string> splitseq(std::vector<Label> seq);

};


#endif //TGKERNEL_TEMPORALGRAPHKERNELAPPROX_H
