/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "TemporalGraphs.h"
#include <iostream>
#include <string>
#include <map>


using namespace std;

void TemporalGraphStream::make_simple() {
    TemporalEdges tes;
    unsigned int num_multiedges = 0;
    for (size_t i = 0; i < edges.size(); ++i){
        TemporalEdge &e = edges.at(i);
        bool found = false;
        for (size_t j = i + 1; j < edges.size(); ++j){
            TemporalEdge &f = edges.at(j);
            if (e.u_id == f.u_id && e.v_id == f.v_id && e.t == f.t) {
                found = true;
                ++num_multiedges;
                break;
            }
        }
        if (!found) {
            tes.push_back(e);
        }
    }
    edges.clear();
    edges.insert(edges.begin(), tes.begin(), tes.end());
}


TemporalGraph TemporalGraphStream::toTemporalGraph() {
    TemporalGraph g;

    for (unsigned int i = 0; i < get_num_nodes(); ++i) {
        TGNode node;
        node.id = i;
        if (timed_node_labels.size() > i)
            node.timed_labels = timed_node_labels.at(i);
        if (!static_node_labels.empty()) {
            node.static_label = static_node_labels.at(i);
        }
        g.nodes.push_back(node);
        g.num_nodes++;
    }

    for (TemporalEdge e : edges) {
        g.nodes.at(e.u_id).adjlist.push_back(e);
        g.num_edges++;
    }

    g.max_time = get_max_time();
    g.label = label;

    return g;
}


void TemporalGraphStream::removeNonConnectecVertices() {
    NodeId nid = 0;
    vector<long> usedVertices(get_num_nodes(), -1);
    for (TemporalEdge &e : edges) {
        if (usedVertices.at(e.u_id) == -1) {
            usedVertices.at(e.u_id) = nid++;
        }
        if (usedVertices.at(e.v_id) == -1) {
            usedVertices.at(e.v_id) = nid++;
        }
    }

    TimedLabels vlabels;
    for (size_t i = 0; i < usedVertices.size(); ++i) {
        long n = usedVertices.at(i);
        if (n != -1) {
            vlabels.push_back(timed_node_labels.at(i));
        }
    }
    for (TemporalEdge &e : edges) {
        e.u_id = (NodeId) usedVertices.at(e.u_id);
        e.v_id = (NodeId) usedVertices.at(e.v_id);
    }

    timed_node_labels.clear();
    timed_node_labels.insert(timed_node_labels.begin(), vlabels.begin(), vlabels.end());
}

TemporalEdges TemporalGraphStream::reduce() {
    TemporalEdges tes;
    std::stable_sort(edges.begin(), edges.end(), [](const TemporalEdge & a, const TemporalEdge & b) -> bool {
        return a.v_id < b.v_id;
    });
    std::stable_sort(edges.begin(), edges.end(), [](const TemporalEdge & a, const TemporalEdge & b) -> bool {
        return a.u_id < b.u_id;
    });
    TemporalEdge le;
    for (TemporalEdge &e : edges) {
        if (e.v_id != le.v_id || e.u_id != le.u_id) {
            le = e;
            tes.push_back(e);
        }
    }
    std::stable_sort(tes.begin(), tes.end(), [](const TemporalEdge & a, const TemporalEdge & b) -> bool {
        return a.t < b.t;
    });
    Time t = 0;
    Time lt = 0;
    for (TemporalEdge &e : tes) {
        if (e.t != lt) {
            ++t;
            lt = e.t;
        }
        e.t = t;
    }
    return tes;
}

TemporalEdges TemporalGraphStream::reSequenceTimes() {
    TemporalEdges tes = edges;
    std::stable_sort(tes.begin(), tes.end(), [](const TemporalEdge & a, const TemporalEdge & b) -> bool {
        return a.t < b.t;
    });
    Time t = 0;
    Time lt = 0;
    for (TemporalEdge &e : tes) {
        if (e.t != lt) {
            ++t;
            lt = e.t;
        }
        e.t = t;
    }
    return tes;
}


void TemporalGraphStream::sort_edges() {
    std::sort(edges.begin(), edges.end(), [](const TemporalEdge & a, const TemporalEdge & b) -> bool { return a.t < b.t; });
}


Labels TemporalGraphStream::getFlattenedNodeLabels() {
    Labels result;
    for (auto &l : timed_node_labels) {
        Label fl = 0;
        for (auto &time_label : l){
            fl = max(fl, time_label.second);
        }
        result.push_back(fl);
    }
    return result;
}

Labels TemporalGraphStream::getFirstNodeLabels() {
    Labels result;
    for (auto &l : timed_node_labels) {
        result.push_back(l.at(0).second);
    }
    return result;
}

Labels TemporalGraphStream::getCombinedNodeLabels() {
    Labels result;
    for (auto &l : timed_node_labels) {
        Label fl = 0;
        unsigned int fac = 1;
        for (size_t i = 0; i < l.size(); i++) {
            auto &time_label = l.at(i);
            fl += fac * (time_label.second + 1);
            fac *= 10;
        }
        result.push_back(fl);
    }
    return result;
}

Labels TemporalGraphStream::getOrderedFlattenedNodeLabels() {
    vector<Time> times;
    for (auto &l : timed_node_labels) {
        Time t = 0;
        for (auto &time_label : l){
            if (time_label.second > 0) {
                t = time_label.first;
                break;
            }
        }
        times.push_back(t);
    }
    Labels result;
    map<Time, Label > label_map;
    map<Time, Label>::iterator it;
    Label l = 1;
    label_map.emplace(0, 0);
    for (Time t : times) {
        it = label_map.find(t);
        if (it == label_map.end()) {
            label_map.emplace(t, l);
            result.push_back(l++);
        } else {
            result.push_back(it->second);
        }
    }
    return result;
}


Label TemporalGraphStream::getNodeLabelAtTime(NodeId nid, Time time) {
    Label result = timed_node_labels.at(nid).at(0).second;
    for (auto &l : timed_node_labels.at(nid)) {
        if (l.first > time) break;
        result = l.second;
    }
    return result;
}


std::ostream& operator<<(std::ostream& os, const TemporalGraphStream& tgs) {
    std::string s1 = "num_nodes\t" + std::to_string(tgs.get_num_nodes()) + "\n";
    std::string s2 = "num_edges\t" + std::to_string(tgs.edges.size()) + "\n";
    std::string s3 = "max_time\t" + std::to_string(tgs.get_max_time()) + "\n";

    return os << s1 << s2 << s3;
}

std::ostream& operator<<(std::ostream& os, const TemporalEdge &tgs) {
    std::string s1 = std::to_string(tgs.u_id) + " " + std::to_string(tgs.v_id) + " " + std::to_string(tgs.t) + " " + "\n";

    return os << s1;
}

std::vector<std::pair<Time, NodeId>> TGNode::getOrderedNeighbors() {
    std::vector<std::pair<Time, NodeId>> result;
    for (auto &e : adjlist) {
        result.emplace_back(e.t, e.v_id);
    }
    std::sort(result.begin(), result.end(), [](const std::pair<Time, NodeId> & a, const std::pair<Time, NodeId> & b) -> bool { return a.first < b.second; });
    return result;
}