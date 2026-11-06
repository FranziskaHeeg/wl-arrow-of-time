/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_EXPLICITLINEGRAPHKERNEL_H
#define TGKERNEL_EXPLICITLINEGRAPHKERNEL_H

#include "TemporalGraphs.h"
#include <unordered_map>

using Histogram = std::unordered_map<std::string, unsigned int>;

class ExplicitLineGraphKernel {
public:
    void calculaleGramMatrix (TemporalGraphStreams &data, std::string datasetname, unsigned int kmin, unsigned int kmax);

private:

    void getGramMatrix(TemporalGraphStreams &data, std::string datasetname, unsigned int k);

    std::string seqToString(std::vector<Label> seq);

    std::vector<std::vector<Label>> temporalRandomWalkVertexEdgeLabel(TemporalGraph &tg, NodeId s, unsigned int k);

    std::vector<std::vector<Label>> temporalRandomWalkVertexEdgeLabelIt(TemporalGraph &tg, NodeId s, unsigned int k);

    void randomWalkVertexEdgeLabelRek(std::vector<std::vector<Label>> &seqs, std::vector<Label> seq, TemporalGraph &tg, NodeId s, unsigned int k, Time curtime);

    void getS(TemporalGraphStream &tgs, std::vector<NodeId> &S);


};


#endif //TGKERNEL_EXPLICITLINEGRAPHKERNEL_H
