/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include <map>
#include "StaticExpansionKernel.h"
#include <Eigen/LU>
#include "GraphKernelCommonFunctions.h"
#include "wlkernel/ColorRefinementKernel.h"
#include "wlkernel/Graph.h"
#include "Timer.h"
#include "TGRepresentations.h"
#include "ExplicitRW.h"

using namespace std;
using namespace Eigen;


void StaticExpansionKernel::getStaticRepresentation(TemporalGraphStreams &data, vector<MatrixXi> &data_matrices,
                                                    TemporalGraphStreams &data_tgss, bool useLineGraphRepresentation) {
    TGRepresentations tgrep;
    if (useLineGraphRepresentation) {
        tgrep.getDirectedLineGraphRepresentation(data, data_matrices, data_tgss, false);
    } else {
        tgrep.getStaticExpansions(data, data_matrices, data_tgss);
    }
}


void StaticExpansionKernel::calculaleGramMatrix(TemporalGraphStreams &data, string datasetname, KernelType kernelType,
        unsigned int const kmin, unsigned int const kmax, bool useLineGraphRepresentation) {
    Timer timer;
    vector<MatrixXi> data_matrices;
    TemporalGraphStreams data_tgss;
    timer.startTimer();
    getStaticRepresentation(data, data_matrices, data_tgss, useLineGraphRepresentation);
    timer.stopTimer("static_expansion.time");
    MatrixXd nm;
    if (kernelType == WL) {
        for (unsigned int h = kmin; h <= kmax; ++h) {
            MatrixXd m;
            timer.startTimer();
            calculateWL(data_tgss, m, h);
            string fulldatasetname;

            if (useLineGraphRepresentation)
                fulldatasetname = datasetname + "__LGWL_" + to_string(h) +".gram";
            else
                fulldatasetname = datasetname + "__SEWL_" + to_string(h) +".gram";

            timer.stopTimer(fulldatasetname + ".time");
            CommonFunctions::normalizeGramMatrix(m, nm);
            CommonFunctions::writeGramMatrixToFile(nm, data_tgss, fulldatasetname);
        }
    } else {
        ExplicitRW explicitRW;

        string fulldatasetname;
        if (useLineGraphRepresentation) {
            fulldatasetname = datasetname + "__LGKS_";
        } else {
            fulldatasetname = datasetname + "__SEKS_";
        }
        explicitRW.calculaleGramMatrix(data_tgss, fulldatasetname, kmin, kmax);

    }
}


double StaticExpansionKernel::kStepRandomWalkKernel(MatrixXi &m1, TemporalGraphStream &tgs1, MatrixXi &m2,
                                                     TemporalGraphStream &tgs2, unsigned int k_max) {

    // map each product (v_1, v_2) of vertices to a number H(v_1, v_2)
    MatrixXi H(tgs1.get_num_nodes(), tgs2.get_num_nodes());
    H.setZero();

    int n_vx = CommonFunctions::productMapping(tgs1.static_node_labels, tgs2.static_node_labels, H);
    SparseMatrix<double> E(n_vx, n_vx);
    E.setZero();

    // prepare identity matrix
    SparseMatrix<double> I(n_vx, n_vx);
    I.setIdentity();

    // compute the adjacency matrix Ax of the direct product graph
    SparseMatrix<double> Ax(n_vx, n_vx);
    productAdjacency(m1, m2, tgs1.static_node_labels, tgs2.static_node_labels, H, Ax);

    // compute products until k
    SparseMatrix<double> Ax_pow = I;
    SparseMatrix<double> Sum = I;
    for (size_t k = 1; k <= k_max; ++k) {
        Ax_pow = Ax * Ax_pow;
        Sum += Ax_pow;
    }

    // compute the total sum
    double K = 0;
    for (int i = 0; i < Sum.outerSize(); ++i) {
        for (SparseMatrix<double>::InnerIterator it(Sum, i); it; ++it) {
            K += it.value();
        }
    }

    return K;
}


// compute the adjacency matrix Ax of the direct product graph (sparse)
void StaticExpansionKernel::productAdjacency(MatrixXi& e1, MatrixXi& e2, Labels& v1_label, Labels& v2_label, MatrixXi& H, SparseMatrix<double>& Ax) {
    using T = Eigen::Triplet<double>;
    vector<T> v;
    for (int i = 0; i < e1.rows(); i++) {
        for (int j = 0; j < e2.rows(); j++) {
            if (   v1_label[e1(i, 0)] == v2_label[e2(j, 0)]
                && v1_label[e1(i, 1)] == v2_label[e2(j, 1)]
                         && e1(i, 2)  ==          e2(j, 2)
                   ) {
                v.push_back(T(H(e1(i, 0), e2(j, 0)), H(e1(i, 1), e2(j, 1)), 1.0));
            }
        }
    }
    Ax.setFromTriplets(v.begin(), v.end());
}


void StaticExpansionKernel::calculateWL(TemporalGraphStreams &data_tgss, MatrixXd& m, unsigned int h) {
    GraphDatabase gdb;

    for (TemporalGraphStream &tg : data_tgss) {
        EdgeList el;

        EdgeLabels elabels;
        for (TemporalEdge const &e : tg.edges) {
            Edge wle(e.u_id, e.v_id);
            el.push_back(wle);
            elabels.insert({wle, e.t});
        }

        Graph g(true, tg.get_num_nodes(), el, tg.static_node_labels);
        g.set_edge_labels(elabels);
        gdb.push_back(g);
    }

    ColorRefinement::ColorRefinementKernel cr(gdb);
    GramMatrix gm = cr.compute_gram_matrix(h, true, true);
    m = MatrixXd(gm);
}
