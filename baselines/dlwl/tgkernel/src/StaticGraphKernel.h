/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_STATICGRAPHKERNEL_H
#define TGKERNEL_STATICGRAPHKERNEL_H

#include "TemporalGraphs.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

class StaticGraphKernel {

public:

    void calculaleGramMatrix(TemporalGraphStreams &data, std::string datasetname, bool useWL, unsigned int const kmin, unsigned int const kmax);

private:

    void calculateWL(TemporalGraphStreams &data_tgss, Eigen::MatrixXd& m, unsigned int h);

    void temporalGraphStreamToEigenMatrix(TemporalGraphStream &tgs, Eigen::MatrixXi &m);

    unsigned long kStepRandomWalkKernel(Eigen::MatrixXi &m1, TemporalGraphStream &tgs1, Eigen::MatrixXi &m2, TemporalGraphStream &tgs2, Time max_time);

    void productAdjacency(Eigen::MatrixXi& e1, Eigen::MatrixXi& e2, Labels &v1_label, Labels &v2_label, Eigen::MatrixXi& H, Eigen::SparseMatrix<unsigned long>& Ax);
};


#endif //TGKERNEL_NONTEMPORALGRAPHKERNEL_H
