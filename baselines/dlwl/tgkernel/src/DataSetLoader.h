/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_DATAMANAGER_H
#define TGKERNEL_DATAMANAGER_H

#include <string>
#include <vector>
#include <fstream>
#include "TemporalGraphs.h"

class DataManager {

public:

    std::string loadDataSet(std::string data_set_name);

    void getGraph(unsigned int graph_num, std::vector<std::vector<unsigned int>> &graph_node_labels,
                  std::vector<std::vector<unsigned int>> &graph_edge_labels, std::vector<std::pair<unsigned int, unsigned int>> &graph_edges,
                  long &graph_label);

    void saveGraphs(TemporalGraphs &tgs, std::string const &data_set_name);

    std::vector<unsigned int> node_graph_map;

//    std::vector<unsigned int> node_labels;

    std::vector<std::vector<unsigned int>> edge_labels;

    std::vector<std::vector<unsigned int>> node_timed_labels;

    std::vector<long> graph_labels;

    std::vector<std::pair<unsigned int, unsigned int>> edges;

    std::vector<unsigned int> graph_offsets;

    unsigned int num_graphs;

private:

    void loadNodeGraphMap(std::string const &filename);

    void loadNodeLabels(std::string const &filename);

    void loadEdgeLabels(std::string const &filename, bool shift);

    void loadEdges(std::string const &filename);

    void loadGraphLabels(std::string const &filename);

    void calculateGraphOffsets();

};


#endif //TGKERNEL_DATAMANAGER_H
