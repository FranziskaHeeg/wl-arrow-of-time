/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include <iostream>
#include <sstream>
#include "DataSetLoader.h"
#include "TemporalGraphs.h"
#include <limits>
#include <cmath>

using namespace std;

string getDataSetName(string path) {
    std::size_t botDirPos = path.find_last_of("/");
    return path.substr(botDirPos + 1, path.length());
}


void getFileStream(string filename, ifstream &fs) {
    fs.open(filename);
    if (!fs.is_open()) {
        cout << "Could not open data set " << filename << endl;
        exit(EXIT_FAILURE);
    }
}

vector<unsigned int> split_string(string s) {
    vector<unsigned int> result;
    stringstream ss(s);
    while (ss.good()) {
        string substr;
        getline(ss, substr, ',');
        result.push_back(stoul(substr));
    }
    return result;
}

string DataManager::loadDataSet(std::string const data_set_name) {

    string node_graph_map_file = data_set_name + "_graph_indicator.txt";
    string node_labels_file = data_set_name + "_node_labels.txt";
    string edge_attributes_file = data_set_name + "_edge_attributes.txt";
    string edges_file = data_set_name + "_A.txt";
    string graph_labels_file = data_set_name + "_graph_labels.txt";

    loadNodeGraphMap(node_graph_map_file);
    loadNodeLabels(node_labels_file);
    loadEdgeLabels(edge_attributes_file, false);
    loadEdges(edges_file);
    loadGraphLabels(graph_labels_file);
    calculateGraphOffsets();

    return getDataSetName(data_set_name);
}

void DataManager::loadNodeGraphMap(std::string const &filename) {
    ifstream fs;
    getFileStream(filename, fs);
    string line;
    while (getline(fs, line)) {
        node_graph_map.push_back(stoi(line) - 1);
    }
    fs.close();
    num_graphs = node_graph_map.back() + 1;
}

void DataManager::loadNodeLabels(std::string const &filename) {
    ifstream fs;
    getFileStream(filename, fs);
    string line;
    while (getline(fs, line)) {
        vector<unsigned int> val = split_string(line);
//        for (unsigned int i : val)
//            cout << i << " ";
//        cout << endl;
        node_timed_labels.push_back(val);
    }
    fs.close();
}

void DataManager::loadEdgeLabels(std::string const &filename, bool shift) {
    ifstream fs;
    getFileStream(filename, fs);
    string line;
    unsigned int minval = std::numeric_limits<unsigned int>::max();
    while (getline(fs, line)) {
        vector<unsigned int> val = split_string(line);
        edge_labels.push_back(val);

        for (unsigned int v : val) {
            if (v < minval)
                minval = v;
        }
    }

    if (shift) {
        for (vector<unsigned int> &els : edge_labels) {
            for (unsigned int &i : els)
                i -= (minval - 1);
        }
    }
    fs.close();
}

void DataManager::loadEdges(std::string const &filename) {
    ifstream fs;
    getFileStream(filename, fs);
    string line;
    while (getline(fs, line)) {
        vector<unsigned int> d = split_string(line);
        if (d.size() >= 2)
            edges.push_back(pair<unsigned int, unsigned int>(d[0],d[1]));
    }
    fs.close();
}

void DataManager::loadGraphLabels(std::string const &filename) {
    ifstream fs;
    getFileStream(filename, fs);
    string line;
    while (getline(fs, line)) {
        graph_labels.push_back(stoi(line));
    }
    fs.close();
}

void DataManager::calculateGraphOffsets() {
    int offset = 0;
    graph_offsets.push_back(offset);
    int c = 0;
    while (c < node_graph_map.size()) {
        if (c + 1 < node_graph_map.size()) {
            if (node_graph_map.at(c + 1) != node_graph_map.at(c)) {
                graph_offsets.push_back(c + 1);
            }
        }
        c++;
    }
}

void DataManager::getGraph(unsigned int const graph_num, std::vector<std::vector<unsigned int>> &graph_node_labels,
                           std::vector<std::vector<unsigned int>> &graph_edge_labels, std::vector<std::pair<unsigned int, unsigned int>> &graph_edges,
                           long &graph_label) {

    unsigned int start = graph_offsets.at(graph_num);
    unsigned int num;

    for (num = start; num < node_graph_map.size() && node_graph_map[num] == node_graph_map[start] ; ++num) {
        graph_node_labels.push_back(node_timed_labels[num]);
    }

    vector<unsigned int> edges_pos;
    for (int i = 0; i < edges.size(); ++i) {
        pair<unsigned int, unsigned int> e = edges[i];
        if (e.first > start && e.first <= num) {
            edges_pos.push_back(i);
            graph_edges.push_back(pair<unsigned int, unsigned int>(e.first - start - 1, e.second - start - 1));
        }
    }

    for (unsigned int i : edges_pos) {
        graph_edge_labels.push_back(edge_labels.at(i));
    }

    graph_label = graph_labels.at(graph_num);
}

void DataManager::saveGraphs(TemporalGraphs &tgs, string const &data_set_name) {
    string graph_labels_filename = data_set_name + "_graph_labels.txt";
    string node_graph_map_filename = data_set_name + "_graph_indicator.txt";
    string node_labels_filename = data_set_name + "_node_labels.txt";
    string edge_attributes_filename = data_set_name + "_edge_attributes.txt";
    string edges_filename = data_set_name + "_A.txt";
    string info_filename = data_set_name + "_info.txt";

    ofstream graph_labels_file;
    graph_labels_file.open(graph_labels_filename);

    ofstream node_graph_map_file;
    node_graph_map_file.open(node_graph_map_filename);

    ofstream node_labels_file;
    node_labels_file.open(node_labels_filename);

    ofstream edge_attributes_file;
    edge_attributes_file.open(edge_attributes_filename);

    ofstream edges_file;
    edges_file.open(edges_filename);

    ofstream info_file;
    info_file.open(info_filename);

    unsigned int offset = 1;
    unsigned int graph_num = 1;

    unsigned long max_num_nodes = 0;
    unsigned long min_num_nodes = 10000000;
    double_t avg_num_nodes = 0;

    unsigned long max_num_edges = 0;
    unsigned long min_num_edges = 10000000;
    double_t avg_num_edges = 0;

    unsigned long class_label_1 = 0;

    for (TemporalGraph &tg : tgs) {
        if (tg.nodes.size() == 0) continue;

        graph_labels_file << tg.label << endl;

        unsigned int num_edges = 0;
        for (TGNode &n : tg.nodes) {
//            node_labels_file << n.label << endl;
            int c = 0;
            node_labels_file << n.static_label << endl;
//            for (std::pair<Time, Label > &p : n.timed_labels) {
//                if (c < n.timed_labels.size() - 1)
//                    node_labels_file << p.first << ", " << p.second << ", ";
//                else
//                    node_labels_file << p.first << ", " << p.second << endl;
//                ++c;
//            }

            node_graph_map_file << graph_num << endl;

            for (TemporalEdge &e : n.adjlist) {
                edges_file << (e.u_id + offset) << ", " << (e.v_id + offset) << endl;
                edge_attributes_file << e.t << endl;
//                edge_attributes_file << e.t << ", " << e.new_label_v << endl;
            }

            num_edges += n.adjlist.size();
        }

        offset += tg.nodes.size();
        ++graph_num;

        if (num_edges < min_num_edges) min_num_edges = num_edges;
        if (num_edges > max_num_edges) max_num_edges = num_edges;
        avg_num_edges += num_edges;

        if (tg.nodes.size() < min_num_nodes) min_num_nodes = tg.nodes.size();
        if (tg.nodes.size() > max_num_nodes) max_num_nodes = tg.nodes.size();
        avg_num_nodes += tg.nodes.size();

        if (tg.label == 1) class_label_1++;
    }

    info_file << "# graphs: " << tgs.size() << endl;
    info_file << "# class 0: " << (tgs.size() - class_label_1) << endl;
    info_file << "# class 1: " << class_label_1 << endl;

    info_file << "min. # nodes: " << min_num_nodes << endl;
    info_file << "max. # nodes: " << max_num_nodes << endl;
    info_file << "avg. # nodes: " << (avg_num_nodes / tgs.size()) << endl;

    info_file << "min. # edges: " << min_num_edges << endl;
    info_file << "max. # edges: " << max_num_edges << endl;
    info_file << "avg. # edges: " << (avg_num_edges / tgs.size()) << endl;

    graph_labels_file.close();
    node_graph_map_file.close();
    node_labels_file.close();
    edge_attributes_file.close();
    edges_file.close();
    info_file.close();
}