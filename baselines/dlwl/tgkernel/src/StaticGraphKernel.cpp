/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include <fstream>
#include "StaticGraphKernel.h"
#include "GraphKernelCommonFunctions.h"
#include "wlkernel/ColorRefinementKernel.h"
#include "wlkernel/Graph.h"
#include "Timer.h"
#include "ExplicitRW.h"

using namespace Eigen;
using namespace std;


void StaticGraphKernel::calculaleGramMatrix(TemporalGraphStreams &data, string datasetname, bool useWL, unsigned int const kmin, unsigned int const kmax) {
    Timer timer;
    if (useWL) {
        for (unsigned int h = kmin; h <= kmax; ++h) {
            MatrixXd nm;
            MatrixXd m;
            string fulldatasetname = datasetname + "__STWL_" + to_string(h) +".gram";
            timer.startTimer();
            calculateWL(data, m, h);
            timer.stopTimer(fulldatasetname + ".time");
            CommonFunctions::normalizeGramMatrix(m, nm);
            CommonFunctions::writeGramMatrixToFile(nm, data, fulldatasetname);
        }
    } else {
        for (TemporalGraphStream &tgs : data) {
            tgs.edges = tgs.reduce();
            tgs.static_node_labels = tgs.getOrderedFlattenedNodeLabels();
        }
        ExplicitRW explicitRW;
        string fulldatasetname = datasetname + "__STRW_";
        explicitRW.calculaleGramMatrix(data, fulldatasetname, kmin, kmax);
    }
}


void StaticGraphKernel::temporalGraphStreamToEigenMatrix(TemporalGraphStream &tgs, MatrixXi &m) {
    TemporalEdges tes = tgs.reduce();
    m.resize(tes.size(), 3);
    for (unsigned int i = 0; i < tes.size(); i++) {
        m(i, 0) = tes.at(i).u_id;
        m(i, 1) = tes.at(i).v_id;
        m(i, 2) = tes.at(i).t;
    }
}

unsigned long StaticGraphKernel::kStepRandomWalkKernel(MatrixXi &m1, TemporalGraphStream &tgs1, MatrixXi &m2,
                                                     TemporalGraphStream &tgs2, Time max_time) {

    // map each product (v_1, v_2) of vertices to a number H(v_1, v_2)
    MatrixXi H(tgs1.get_num_nodes(), tgs2.get_num_nodes());
    H.setZero();


    Labels tg1_flat_labels = tgs1.getOrderedFlattenedNodeLabels();
    Labels tg2_flat_labels = tgs2.getOrderedFlattenedNodeLabels();

    int n_vx = CommonFunctions::productMapping(tg1_flat_labels, tg2_flat_labels, H);
    SparseMatrix<unsigned long> E(n_vx, n_vx);
    E.setZero();
    SparseMatrix<unsigned long> Sum = E;

    // prepare identity matrix
    SparseMatrix<unsigned long> I(n_vx, n_vx);
    I.setIdentity();

    // compute the adjacency matrix Ax of the direct product graph
    SparseMatrix<unsigned long> Ax(n_vx, n_vx);
    productAdjacency(m1, m2, tg1_flat_labels, tg2_flat_labels, H, Ax);

    // compute products until k
    int k_max = (int) max_time;
    SparseMatrix<unsigned long> Ax_pow = I;
    for (int k = 1; k <= k_max; k++) {
        Ax_pow = Ax * Ax_pow;
        Sum += Ax_pow;
    }

    // compute the total sum
    unsigned long K = 0;
    for (int i = 0; i < Sum.outerSize(); ++i) {
        for (SparseMatrix<unsigned long>::InnerIterator it(Sum, i); it; ++it) {
            K += it.value();
        }
    }

    return K + n_vx;
}


void StaticGraphKernel::productAdjacency(MatrixXi &e1, MatrixXi &e2, Labels &v1_label, Labels &v2_label, MatrixXi& H, SparseMatrix<unsigned long>& Ax) {
    using T = Eigen::Triplet<unsigned long>;

    vector<T> v;
    for (int i = 0; i < e1.rows(); i++) {
        for (int j = 0; j < e2.rows(); j++) {
            if (   v1_label[e1(i, 0)] == v2_label[e2(j, 0)]
                   && v1_label[e1(i, 1)] == v2_label[e2(j, 1)]
                   && e1(i, 2) == e2(j, 2)
                   ) {
                v.push_back(T(H(e1(i, 0), e2(j, 0)), H(e1(i, 1), e2(j, 1)), 1.0));
            }
        }
    }

    Ax.setFromTriplets(v.begin(), v.end());
}


void StaticGraphKernel::calculateWL(TemporalGraphStreams &data_tgss, MatrixXd& m, unsigned int h) {
    GraphDatabase gdb;

    vector<Edge> es;

    for (TemporalGraphStream &tg : data_tgss) {
        EdgeList el;

        TemporalEdges tes = tg.reduce();

        EdgeLabels elabels;
        for (TemporalEdge &e : tes) {
            Edge wle(e.u_id, e.v_id);
            el.push_back(wle);
            elabels.insert({wle, e.t});
        }


//        Graph g(false, tg.get_num_nodes(), el, tg.getCombinedNodeLabels());
        Graph g(false, tg.get_num_nodes(), el, tg.getOrderedFlattenedNodeLabels());
        g.set_edge_labels(elabels);
        gdb.push_back(g);
    }

    ColorRefinement::ColorRefinementKernel cr(gdb);
    GramMatrix gm = cr.compute_gram_matrix(h, true, true);
    m = MatrixXd(gm);
}
