/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_GRAPHDATALOADER_H
#define TGKERNEL_GRAPHDATALOADER_H

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using NodeId = unsigned int;
using Time = unsigned long;
using Cost = float;
using Label = long;
using Labels = std::vector<Label>;
using TimedLabelVec = std::vector<std::pair<Time, Label >>;
using TimedLabels = std::vector<std::vector<std::pair<Time, Label >>>;

struct TemporalEdge;
using AdjacenceList = std::vector<TemporalEdge>;

struct TGNode {
    NodeId id;
    AdjacenceList adjlist;
    Label static_label;
    TimedLabelVec timed_labels;

    Label getLabel(Time t) {
        Label result = timed_labels.at(0).second;
        for (auto &l : timed_labels) {
            if (l.first > t) break;
            result = l.second;
        }
        return result;
    }

    std::vector<std::pair<Time, NodeId>> getOrderedNeighbors();
};

struct TemporalEdge {
    TemporalEdge() {};
    TemporalEdge(NodeId u, NodeId v, Time time, Cost c, Time tt) : u_id(u), v_id(v), t(time), cost(c), traversal_time(tt) {};
    TemporalEdge(NodeId u, NodeId v, Time time) :
            u_id(u), v_id(v), t(time), cost(0), traversal_time(0) {};
    NodeId u_id, v_id;
    Time t;
    Cost cost;
    Time traversal_time;
};


using TGNodes = std::vector<TGNode>;
using TemporalEdges = std::vector<TemporalEdge>;

struct TemporalGraph {
    TemporalGraph () : num_nodes(0), num_edges(0), max_time(0), label(0), nodes(TGNodes()) {};
    unsigned int num_nodes;
    unsigned int num_edges;
    Time max_time;
    Label label;
    TGNodes nodes;
};

using TemporalGraphs = std::vector<TemporalGraph>;

struct TemporalGraphStream {
    Label label;
    TemporalEdges edges;
    Labels static_node_labels;
    TimedLabels timed_node_labels;

    Time get_max_time() const {
        if (edges.size() == 0) return 0;
        return edges.back().t;
    }

    unsigned long get_num_nodes() const {
        if (!static_node_labels.empty()) return static_node_labels.size();
        return timed_node_labels.size();
    }

    // create single layer simple graph
    TemporalEdges reduce();

    // replace times with sequence positions
    TemporalEdges reSequenceTimes();

    // remove non-connected vertices
    void removeNonConnectecVertices();

    void sort_edges();

    TemporalGraph toTemporalGraph();

    void make_simple();

    Labels getFlattenedNodeLabels();

    Label getNodeLabelAtTime(NodeId, Time);

    Labels getOrderedFlattenedNodeLabels();

    Labels getFirstNodeLabels();

    Labels getCombinedNodeLabels();
};


//using TemporalGraphs = std::vector<TemporalGraph>;
using TemporalGraphStreams = std::vector<TemporalGraphStream>;


std::ostream& operator<<(std::ostream& os, const TemporalGraphStream& tgs);
std::ostream& operator<<(std::ostream& os, const TemporalEdge &tgs);


#endif //TGKERNEL_GRAPHDATALOADER_H
