/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include <iostream>
#include "DataSetLoader.h"
#include "TemporalGraphs.h"
#include <string>
#include <algorithm>
#include "StaticGraphKernel.h"
#include "NonTemporalGraphKernel.h"
#include "StaticExpansionKernel.h"
#include "Timer.h"
#include <chrono>
#include "TGStatistics.h"
#include "TemporalGraphKernelApprox.h"
#include "ExplicitLineGraphKernel.h"
#include "TGRepresentations.h"
#include <Eigen/LU>

using namespace std;

/**
 * loads dataset and constructs temporal graph streams
 */
string constructTGStreams(string const dataset, TemporalGraphStreams &tgstreams, unsigned int const min_nodes, unsigned int const min_edges) {
    DataManager dm;
	string datasetname = dm.loadDataSet(dataset);

    for (unsigned int graph_num = 0; graph_num < dm.num_graphs; graph_num++) {

        cout << "Graph " << graph_num << ":\t";

        std::vector<std::vector<unsigned int>> graph_node_labels;
		std::vector<std::vector<unsigned int>> graph_edge_labels;
        std::vector<std::pair<unsigned int, unsigned int>> graph_edges;
		Label graph_label;

        dm.getGraph(graph_num, graph_node_labels, graph_edge_labels, graph_edges, graph_label);

        if (graph_node_labels.empty() || graph_edges.empty()) continue;

        TemporalGraphStream tgs;

		for (unsigned int i = 0; i < graph_edge_labels.size() && i < graph_edges.size(); ++i) {
			auto e = graph_edges[i];

			for (unsigned long label : graph_edge_labels[i]) {

				TemporalEdge tedge;
				tedge.u_id = e.first;
				tedge.v_id = e.second;
				tedge.t = label;

				tgs.edges.push_back(tedge);
			}
		}

		tgs.sort_edges();
        for (size_t i = 0; i < graph_node_labels.size(); ++i) {
            TimedLabelVec tl;
            for (size_t j = 0; j < graph_node_labels.at(i).size() - 1; j += 2) {
                tl.push_back({graph_node_labels.at(i).at(j), graph_node_labels.at(i).at(j + 1)});
            }
            tgs.timed_node_labels.push_back(tl);
        }

		tgs.label = graph_label;

        if (tgs.edges.size() >= min_edges && tgs.get_num_nodes() >= min_nodes) {
			tgstreams.push_back(tgs);
			cout << tgs.get_num_nodes() << " nodes\t" << tgs.edges.size() << " edges" << endl << flush;
		} else {
            cout << tgs.get_num_nodes() << " nodes\t" << tgs.edges.size() << " edges"  << "...too small "<< endl << flush;
        }

    }
    cout << "\nDataset name: " << datasetname << endl << flush;
    cout << "Size of dataset: " << tgstreams.size() << " graphs\n" << endl << flush;
	return datasetname;
}



int main(int argc, char *argv[]) {
    std::srand(time(NULL));

	if (argc < 5) {
		cout << "Please provide path to dataset, action [0-13], kmin and kmax" << endl;
		cout << endl << "Actions:" << endl;
        cout << "0   Print statistics" << endl;
        cout << "1   Non-temporal WL kernel" << endl;
        cout << "2   Non-temporal kstep-RW kernel" << endl;
        cout << "4   Reduced TG WL kernel" << endl;
        cout << "5   Reduced TG kstep-RW kernel" << endl;
        cout << "7   Line graph expansion WL kernel" << endl;
        cout << "8   Line graph expansion kstep-RW kernel" << endl;
        cout << "9   Explicit line graph expansion kstep-RW kernel" << endl;
        cout << "10  Static expansion WL kernel" << endl;
        cout << "11  Static expansion kstep-RW kernel" << endl;
        cout << "13  Temporal kstep-RW kernel (approx)" << endl;
        cout << "15  Export transformed static expansions" << endl;
        cout << "16  Export transformed line graphs" << endl;
        cout << "17  Export transformed reduced graphs" << endl;
        cout << "18  Export transformed baseline graphs" << endl;
        cout << endl << "Usage: tgkernel <data set> <action> <kmin> <kmax>" << endl;
		exit(1);
	}

	TemporalGraphStreams tgss;

	string datasetname = constructTGStreams(argv[1], tgss, 2, 1);

    StaticGraphKernel staticgk;
    NonTemporalGraphKernel nontgk;
    StaticExpansionKernel setgk;

    unsigned int const kmin = stoi(argv[3]);
    unsigned int const kmax = stoi(argv[4]);

    switch(stoi(argv[2])) {
        case 0 : {
            TGStatistics tgstats;
            tgstats.getTemporalGraphStreamsStatistics(tgss);
            break;
        }
        case 1 : {
            cout << "Non-temporal WL kernel" << flush;
            try {
                nontgk.calculaleGramMatrix(tgss, datasetname, true, kmin, kmax);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 2 : {
            cout << "Non-temporal kstep-RW kernel" << flush;
            try {
                nontgk.calculaleGramMatrix(tgss, datasetname, false, kmin, kmax);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }

        case 4 : {
            cout << "Reduced TG WL kernel" << flush;
            try {
                staticgk.calculaleGramMatrix(tgss, datasetname, true, kmin, kmax);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 5 : {
            cout << "Reduced TG kstep-RW kernel" << flush;
            try {
                staticgk.calculaleGramMatrix(tgss, datasetname, false, kmin, kmax);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 7 : {
            cout << "Line graph expansion WL kernel" << flush;
            try {
                setgk.calculaleGramMatrix(tgss, datasetname, StaticExpansionKernel::KernelType::WL, kmin, kmax, true);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 8 : {
            cout << "Line graph expansion kstep-RW kernel" << flush;
            try {
                setgk.calculaleGramMatrix(tgss, datasetname, StaticExpansionKernel::KernelType::KSTEP, kmin, kmax-1, true);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 9 : {
            cout << "Explicit line graph expansion kstep-RW kernel" << flush;
            try {
                ExplicitLineGraphKernel explg;
                string fulldatasetname = datasetname + "__LDEX_";
                explg.calculaleGramMatrix(tgss, fulldatasetname, kmin, kmax);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 10 : {
            cout << "Static expansion WL kernel" << flush;
            try {
                setgk.calculaleGramMatrix(tgss, datasetname, StaticExpansionKernel::KernelType::WL, kmin, kmax, false);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 11 : {
            cout << "Static expansion kstep-RW kernel" << flush;
            try {
                setgk.calculaleGramMatrix(tgss, datasetname, StaticExpansionKernel::KernelType::KSTEP, kmin, kmax, false);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
        case 13 : {
            cout << "Temporal kstep approx" << flush;
            try {
                int S_size = 1000;
                if (argc > 3) {
                    S_size = stoi(argv[5]);
                }
                TemporalGraphKernelApprox tgappr;
                string fulldatasetname = datasetname + "__TMAP" + to_string(S_size) + "_";
                tgappr.calculaleGramMatrix(tgss, fulldatasetname, kmin, kmax, S_size);
                cout << "done" << endl;
            } catch (Timeout &) {
                cout << "TIMEOUT\n" << flush;
            }
            break;
        }
//        case 14 : {
//            cout << "Temporal WL" << flush;
//            try {
//                string fulldatasetname = datasetname + "__TWWW_" + to_string(kmin) + ".gram";
//                TemporalGraphs graphs;
//                for (auto &tgs : tgss)
//                    graphs.push_back(tgs.toTemporalGraph());
//                Todo twlk(graphs);
//                GramMatrix gm = twlk.compute_gram_matrix(kmin, true, false);
//                Eigen::MatrixXd m = Eigen::MatrixXd(gm);
//                Eigen::MatrixXd nm;
//                CommonFunctions::normalizeGramMatrix(m, nm);
//                CommonFunctions::writeGramMatrixToFile(nm, tgss, fulldatasetname);
//                cout << "done" << endl;
//            } catch (Timeout &) {
//                cout << "TIMEOUT\n" << flush;
//            }
//            break;
//        }
        case 15 : {
            Timer timer;
            cout << "Transforming to SEs" << endl;
            TGRepresentations tgrep;
            vector<Eigen::MatrixXi> m;
            TemporalGraphStreams static_expansions;

            timer.startTimer();
            tgrep.getStaticExpansions(tgss, m, static_expansions);
            timer.stopTimer(datasetname + "SE.time");

            TemporalGraphs setg;
            for (auto &tgs : static_expansions) {
                setg.push_back(tgs.toTemporalGraph());
            }

            DataManager dsl;
            dsl.saveGraphs(setg, datasetname + "SE");
            cout << "written to " << datasetname + "SE" << endl;
            break;
        }
        case 16 : {
            Timer timer;
            cout << "Transforming to Line graphs" << endl;
            TGRepresentations tgrep;
            vector<Eigen::MatrixXi> m;
            TemporalGraphStreams static_expansions;
            timer.startTimer();
            tgrep.getDirectedLineGraphRepresentation(tgss, m, static_expansions, false);
            timer.stopTimer(datasetname + "LG.time");

            TemporalGraphs setg;
            for (auto &tgs : static_expansions) {
                setg.push_back(tgs.toTemporalGraph());
            }

            DataManager dsl;
            dsl.saveGraphs(setg, datasetname + "LG");
            cout << "written to " << datasetname + "LG" << endl;
            break;
        }
        case 17 : {
            Timer timer;
            cout << "Transforming to Reduced graphs" << endl;
            TemporalGraphs setg;
            TemporalGraphStreams tgss2;
            timer.startTimer();
            for (TemporalGraphStream &tgs : tgss) {
                tgs.edges = tgs.reduce();
//                tgs.static_node_labels = tgs.getCombinedNodeLabels();
                tgs.static_node_labels = tgs.getOrderedFlattenedNodeLabels(); // paper version
                tgss2.push_back(tgs);
            }
            timer.stopTimer(datasetname + "RG.time");

            for (TemporalGraphStream &tgs : tgss2) {
                setg.push_back(tgs.toTemporalGraph());
            }
            DataManager dsl;
            dsl.saveGraphs(setg, datasetname + "RG");
            cout << "written to " << datasetname + "SE" << endl;
            break;
        }
        case 18 : {
            Timer timer;
            cout << "Transforming to Baseline graphs" << endl;
            TemporalGraphs setg;
            TemporalGraphStreams tgss2;
            timer.startTimer();
            for (TemporalGraphStream &tgs : tgss) {
                tgs.static_node_labels = tgs.getCombinedNodeLabels();
                tgss2.push_back(tgs);
            }
            timer.stopTimer(datasetname + "BG.time");

            for (TemporalGraphStream &tgs : tgss2) {
                setg.push_back(tgs.toTemporalGraph());
            }
            DataManager dsl;
            dsl.saveGraphs(setg, datasetname + "BG");
            cout << "written to " << datasetname + "SE" << endl;
            break;
        }
        default:
            cout << "No such mode." << flush;
    }


	return 0;
}
