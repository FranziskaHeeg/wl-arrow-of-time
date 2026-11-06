/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "TGRepresentations.h"
#include <cstdlib>

using namespace std;
using namespace Eigen;

void TGRepresentations::getDirectedLineGraphRepresentation(TemporalGraphStreams &data, vector<MatrixXi> &data_matrices,
                                            TemporalGraphStreams &data_tgss, bool use_edge_labels) {

    unsigned int fac1 = 10;

    for (TemporalGraphStream &tgs : data) {
        TemporalGraphStream ntg;

        temporalGraphStreamToDirectedLineGraph(tgs, ntg, fac1);

        if (ntg.get_num_nodes() == 0 || ntg.edges.empty()) continue;

        data_tgss.push_back(ntg);

        MatrixXi matrix;
        temporalGraphStreamToMatrix(ntg, matrix);
        data_matrices.push_back(matrix);
    }
}

void TGRepresentations::getStaticExpansions(TemporalGraphStreams &data, vector<MatrixXi> &data_matrices,
                                            TemporalGraphStreams &data_tgss) {

    for (TemporalGraphStream &tgs : data) {
        TemporalGraphStream ntg;
        temporalGraphStreamToStaticExpansion(tgs, ntg);

        if (ntg.static_node_labels.empty() || ntg.edges.empty()) continue;

        data_tgss.push_back(ntg);

        MatrixXi matrix;
        temporalGraphStreamToMatrix(ntg, matrix);
        data_matrices.push_back(matrix);
    }
}


void TGRepresentations::temporalGraphStreamToStaticExpansion(TemporalGraphStream tgs, TemporalGraphStream &tgsout) {

    using P = pair<NodeId,Time>;
    map<P, NodeId> nodemap;
    map<P, NodeId>::iterator it;

    map<NodeId, Time> lastNodeTime;
    map<NodeId, Time>::iterator itl;

    NodeId nid = 0;
    vector<pair<P, P>> tn;

    tgs.sort_edges();

    for (auto &e : tgs.edges) {

        it = nodemap.find(P(e.u_id, e.t));
        if (it == nodemap.end()) {
            nodemap.emplace(P(e.u_id, e.t), nid++);
            tgsout.static_node_labels.push_back(tgs.getNodeLabelAtTime(e.u_id, e.t));
        }

        it = nodemap.find(P(e.v_id, e.t + 1));
        if (it == nodemap.end()) {
            nodemap.emplace(P(e.v_id, e.t + 1), nid++);
            tgsout.static_node_labels.push_back(tgs.getNodeLabelAtTime(e.v_id, e.t + 1));
        }

        TemporalEdge x(nodemap.at(P(e.u_id, e.t)), nodemap.at(P(e.v_id, e.t + 1)), 0);
        tgsout.edges.push_back(x);

        //todo check if necessary
//        it = nodemap.find(P(e.u_id, e.t + 1));
//        if (it == nodemap.end()) {
//            nodemap.emplace(P(e.u_id, e.t + 1), nid++);
//            tgsout.node_labels.push_back( tgs.node_labels.at(e.u_id));
//        }

        itl = lastNodeTime.find(e.u_id);
        if (itl == lastNodeTime.end()) {
            lastNodeTime.emplace(e.u_id, e.t);
        } else {
            if (itl->second < e.t) {

                TemporalEdge f(nodemap.at(P(e.u_id, itl->second)), nodemap.at(P(e.u_id, e.t)), 1);
                tgsout.edges.push_back(f);

                TemporalEdge f2(nodemap.at(P(e.u_id, itl->second + 1)), nodemap.at(P(e.u_id, e.t)), 1);
                tgsout.edges.push_back(f2);

                lastNodeTime.at(e.u_id) = e.t;
            }
        }
    }

    tgsout.label = tgs.label;

//    cout << endl;
//    cout << tgs.get_num_nodes() << " " << tgs.edges.size() << endl;
//    cout << tgsout.get_num_nodes() << " " << tgsout.edges.size() << endl << endl;
}

void TGRepresentations::temporalGraphStreamToDirectedLineGraph(TemporalGraphStream tgs, TemporalGraphStream &tgsout,
                                                               unsigned int fac1) {
    struct Node {
        NodeId id;
        AdjacenceList adjlist;
        Time maxtime;
        Label label;
        NodeId uid;
        NodeId vid;
        Time t;
    };

    // optional maximum waiting time delta (environment variable TGKERNEL_MAX_WAITING_TIME): two events are
    // only connected if 1 <= t' - t <= delta; unset means no bound, as in the original implementation
    const char *max_waiting_env = std::getenv("TGKERNEL_MAX_WAITING_TIME");
    const bool bounded = max_waiting_env != nullptr;
    const unsigned long max_waiting = bounded ? std::stoul(max_waiting_env) : 0;

   // create one vertex for each (directed) edge
    vector<Node> nodes;
    NodeId nid = 0;
    for (TemporalEdge &e : tgs.edges) {
        Node n;
        n.id = nid++;
        n.uid = e.u_id;
        n.vid = e.v_id;
        n.t = e.t;

        n.label = fac1 * (1 + tgs.getNodeLabelAtTime(e.u_id, e.t)) + (1 + tgs.getNodeLabelAtTime(e.v_id, e.t + 1));

        nodes.push_back(n);

        tgsout.static_node_labels.push_back(n.label);
    }

    for (Node &n : nodes) {
        for (Node &m : nodes) {
            if (n.vid == m.uid && n.t < m.t && (!bounded || m.t - n.t <= max_waiting)) {
                // edge from n to m
                TemporalEdge e(n.id, m.id, 0);

                if (n.t == m.t - 1) e.t = 1;    // no waiting between edges

                tgsout.edges.push_back(e);
            }
        }
    }

    tgsout.label = tgs.label;
//    tgsout.removeNonConnectecVertices();

//    cout << "tgsout edges: " << tgsout.edges.size() << endl;
//    cout << "tgsout nodes: " << tgsout.get_num_nodes() << endl << endl;
}


//void TGRepresentations::temporalGraphStreamToDirectedLineGraphEdgeLabelled(TemporalGraphStream tgs,
//                                                                           TemporalGraphStream &tgsout, unsigned int fac1, unsigned int fac2) {
//    struct Node {
//        NodeId id;
//        AdjacenceList adjlist;
//        Time maxtime;
//        Label label;
//        NodeId uid;
//        NodeId vid;
//        Time t;
//    };
//
//    // create one vertex for each (directed) edge
//    vector<Node> nodes;
//    NodeId nid = 0;
//    for (TemporalEdge &e : tgs.edges) {
//        Node n;
//        n.id = nid++;
//        n.uid = e.u_id;
//        n.vid = e.v_id;
//        n.t = e.t;
//
//        tgsout.static_node_labels.push_back(0);
//        nodes.push_back(n);
//    }
//
//    // add edges
//    for (Node &n : nodes) {
//        for (Node &m : nodes) {
//            if (n.vid == m.uid && n.t < m.t) {
//                // edge from n to m
//                TemporalEdge e(n.id, m.id, 0);
////                e.t = fac2 * (1 + n.uid) + fac1 * (1 + n.vid) + (1 + m.vid);
////                e.t = fac2 * (1 + tgs.node_labels.at(n.uid)) + fac1 * (1 + tgs.node_labels.at(n.vid)) + (1 + tgs.node_labels.at(m.vid));
//                e.t = fac2 * (1 + tgs.getNodeLabelAtTime(n.uid, n.t)) +
//                        fac1 * (1 + tgs.getNodeLabelAtTime(n.vid, n.t + 1)) +
//                        (1 + tgs.getNodeLabelAtTime(m.vid, m.t + 1));
//                tgsout.edges.push_back(e);
//            }
//        }
//    }
//
//    tgsout.label = tgs.label;
//    tgsout.removeNonConnectecVertices();
//
////    cout << endl;
////    cout << tgs.get_num_nodes() << " " << tgs.edges.size() << endl;
////    cout << tgsout.get_num_nodes() << " " << tgsout.edges.size() << endl << endl;
//}

void TGRepresentations::temporalGraphStreamToMatrix(TemporalGraphStream &tgs, MatrixXi &m) {
    m.resize(tgs.edges.size(), 3);
    int i = 0;
    for (TemporalEdge &e : tgs.edges) {
        m(i, 0) = e.u_id;
        m(i, 1) = e.v_id;
        m(i, 2) = e.t;
        i++;
    }
}