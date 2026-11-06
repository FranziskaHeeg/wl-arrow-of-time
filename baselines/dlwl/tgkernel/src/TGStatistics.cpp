/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "TGStatistics.h"

using namespace std;

void TGStatistics::getTemporalGraphStreamsStatistics(TemporalGraphStreams &tgss) {

    double avg_times = 0.0;
    double avg_nodes = 0.0;
    double avg_edges = 0.0;
    double avg_infected = 0.0;
    double avg_infected_c1 = 0.0;
    double avg_infected_c2 = 0.0;
    
    unsigned int c1_size = 0;
    unsigned int c2_size = 0;

    unsigned int total_num_nodes = 0;

    unsigned int min_edges = 10000000;
    unsigned int max_edges = 0;

//    unsigned max_degree = 0;
    double avg_max_degree = 0.0;

    for (TemporalGraphStream &tgs : tgss) {
        tgs.make_simple();

        unsigned int infected = 0;
        for (Label &l : tgs.getFlattenedNodeLabels()) {
            if (l > 0) {
                ++infected;
            }
        }

//        cout << "# nodes: " << tgs.get_num_nodes() << endl;
//        cout << "# infected nodes: " << infected << endl;
//        cout << "# edges: " << tgs.edges.size() << endl;

        avg_infected += infected;

        if (tgs.label == 0) {
            avg_infected_c1 += infected;
            c1_size += tgs.get_num_nodes();
        } else {
            avg_infected_c2 += infected;
            c2_size += tgs.get_num_nodes();
        }

        total_num_nodes += tgs.get_num_nodes();

        avg_edges += tgs.edges.size() / 2;
        avg_nodes += tgs.get_num_nodes();

        if (tgs.edges.size() > max_edges) max_edges = tgs.edges.size();
        if (tgs.edges.size() < min_edges) min_edges = tgs.edges.size();

        unsigned int times = 0;
        vector<unsigned int> edges_per_layer;
        unsigned int epl = 0;
        Time last = tgs.edges.at(0).t;
        for (TemporalEdge &e : tgs.edges) {
            ++epl;
            if (last != e.t) {
                ++times;
                last = e.t;
                edges_per_layer.push_back(epl);
                epl = 0;
            }
        }
//        cout << "# times: " << times << endl;
        avg_times += times;

        unsigned int degree = 0;
        TemporalGraph tg = tgs.toTemporalGraph();
        for (size_t i = 0; i < tgs.get_num_nodes(); ++i) {
//            cout << tg.nodes.at(i).adjlist.size() << endl;

            if (tg.nodes.at(i).adjlist.size() == 0) continue;

            if (tg.nodes.at(i).adjlist.size() > degree) {
                degree = tg.nodes.at(i).adjlist.size();
            }

            unsigned int times = 1;
            Time last = tg.nodes.at(i).adjlist.at(0).t;
            for (TemporalEdge &e : tg.nodes.at(i).adjlist) {
                if (last != e.t) {
                    ++times;
                    last = e.t;
                }
            }
//            cout << "# times at vertex " << i << "  " << times << endl;
        }
        avg_max_degree += degree;

//        cout << endl;

//        for (unsigned int epl : edges_per_layer) {
//            cout << epl << endl;
//        }
//        cout << endl;

//        for (TemporalEdge &e : tgs.edges) {
//           cout << e;
//        }
    }

    avg_times /= tgss.size();
    avg_nodes /= tgss.size();
    avg_edges /= tgss.size();
    avg_max_degree /= tgss.size();
    avg_infected /= total_num_nodes;
    avg_infected_c1 /= c1_size;
    avg_infected_c2 /= c2_size;

    cout << "# graphs: " << tgss.size() << endl;
    cout << "avg. # nodes: " << avg_nodes << endl;
    cout << "avg. # edges: " << avg_edges << endl;
    cout << "avg. # times: " << avg_times << endl << endl;
    cout << "avg. # infec: " << avg_infected << endl;
    cout << "avg. # infec 1: " << avg_infected_c1 << endl;
    cout << "avg. # infec 2: " << avg_infected_c2 << endl << endl;
    cout << "min. # edges: " << min_edges << endl;
    cout << "max. # edges: " << max_edges << endl;
    cout << "avg. max deg: " << avg_max_degree << endl;

}