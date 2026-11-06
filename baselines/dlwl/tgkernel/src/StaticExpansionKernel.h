/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_STATICEXPANSIONKERNEL_H
#define TGKERNEL_STATICEXPANSIONKERNEL_H

#include "TemporalGraphs.h"
#include <Eigen/Core>
#include <Eigen/Sparse>



class StaticExpansionKernel {

public:

    enum KernelType {
        KSTEP, WL
    };

    void calculaleGramMatrix(TemporalGraphStreams &data, std::string datasetname, KernelType kernelType,
            unsigned int const kmin, unsigned int const kmax, bool useLineGraphRepresentation);

private:

    void getStaticRepresentation(TemporalGraphStreams &data, std::vector<Eigen::MatrixXi> &data_matrices, TemporalGraphStreams &data_tgss, bool useLineGraphRepresentation);

    void calculateWL(TemporalGraphStreams &data_tgss, Eigen::MatrixXd& m, unsigned int h);

    double kStepRandomWalkKernel(Eigen::MatrixXi &m1, TemporalGraphStream &tgs1,
                                 Eigen::MatrixXi &m2, TemporalGraphStream &tgs2, unsigned int k_max);


    void productAdjacency(Eigen::MatrixXi& e1, Eigen::MatrixXi& e2, Labels &v1_label, Labels &v2_label, Eigen::MatrixXi& H, Eigen::SparseMatrix<double>& Ax);

};


#endif //TGKERNEL_STATICEXPANSIONKERNEL_H
