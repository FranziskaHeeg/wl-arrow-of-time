/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_TGREPRESENTATIONS_H
#define TGKERNEL_TGREPRESENTATIONS_H

#include "TemporalGraphs.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

class TGRepresentations {

public:

    void getStaticExpansions(TemporalGraphStreams &data, std::vector<Eigen::MatrixXi> &data_matrices,
            TemporalGraphStreams &data_tgss);

    void getDirectedLineGraphRepresentation(TemporalGraphStreams &data, std::vector<Eigen::MatrixXi> &data_matrices,
                             TemporalGraphStreams &data_tgss, bool use_edge_labels);

    void temporalGraphStreamToStaticExpansion(TemporalGraphStream tgs, TemporalGraphStream &tgsout);

    void temporalGraphStreamToMatrix(TemporalGraphStream &tgs, Eigen::MatrixXi &m);

    void temporalGraphStreamToDirectedLineGraph(TemporalGraphStream tgs, TemporalGraphStream &tgsout, unsigned int fac1);

};


#endif //TGKERNEL_TGREPRESENTATIONS_H
