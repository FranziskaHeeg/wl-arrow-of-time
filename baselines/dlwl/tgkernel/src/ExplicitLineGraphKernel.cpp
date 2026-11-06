/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "ExplicitLineGraphKernel.h"
#include "GraphKernelCommonFunctions.h"
#include <Eigen/Sparse>
#include <iterator>
#include "Timer.h"

using namespace std;

void ExplicitLineGraphKernel::calculaleGramMatrix (TemporalGraphStreams &data, string datasetname,
                                      unsigned int const kmin, unsigned int const kmax) {
    for (unsigned int h = kmin; h <= kmax; ++h) {
        string fdatasetname = datasetname + to_string(h) +".gram";
        getGramMatrix(data, fdatasetname, h);
    }
}

string ExplicitLineGraphKernel::seqToString(vector<Label> seq) {
    std::stringstream result;
    std::copy(seq.begin(), seq.end(), std::ostream_iterator<Label>(result, " "));
    return result.str();
}

void ExplicitLineGraphKernel::getGramMatrix(TemporalGraphStreams &data, std::string datasetname, unsigned int k) {

    Timer timer;

    using S = Eigen::Triplet<unsigned int>;

    vector<Histogram> histograms;
    unsigned int numSeqs = 0;
    map<string, unsigned int> sequenceNumbering;

    timer.startTimer();
    for (TemporalGraphStream &tgs : data) {
        TemporalGraph tg = tgs.toTemporalGraph();
        Histogram histogram;
        vector<NodeId> S;
        getS(tgs, S);

        for (NodeId s : S) {
            vector<vector<Label>> seqs;
//            seqs = temporalRandomWalkVertexEdgeLabel(tg, s, k);
            seqs = temporalRandomWalkVertexEdgeLabelIt(tg, s, k);

            for (vector<Label> &label_seq : seqs) {
                string seqstr = seqToString(label_seq);
                auto it = histogram.find(seqstr);
                if (it == histogram.end()) {
                    histogram.insert({{seqstr, 1}});
                    sequenceNumbering.insert({{seqstr, numSeqs++}});
                } else {
                    histogram.at(seqstr)++;
                }
            }
        }

        histograms.push_back(histogram);
    }

    unsigned long num_graphs = histograms.size();
    vector<S> nonzero_components;

//    for (unsigned int i = 0; i < num_graphs; ++i) {
//        auto c = histograms[i];
//        cout << "Graph " << i << endl;
//        for (auto& it: c) {
//            cout << it.second << "\t" << it.first << endl;
//        }
//        cout << endl;
//    }

    for (unsigned int i = 0; i < num_graphs; ++i) {
        auto c = histograms[i];

        for (const auto &j: c) {
            string key = j.first;
            unsigned int value = j.second;
            unsigned int index = sequenceNumbering.find(key)->second;
            nonzero_components.push_back(S(i, index, value));
        }
    }

    using SpMatrix = Eigen::SparseMatrix<double>;
    using GramMatrix = SpMatrix;

    // Compute Gram matrix.
    GramMatrix feature_vectors(num_graphs, numSeqs);
    feature_vectors.setFromTriplets(nonzero_components.begin(), nonzero_components.end());
    GramMatrix gram_matrix(num_graphs, num_graphs);
    gram_matrix = feature_vectors * feature_vectors.transpose();

    Eigen::MatrixXd dense_gram_matrix(gram_matrix);

    Eigen::MatrixXd nm;
    CommonFunctions::normalizeGramMatrix(dense_gram_matrix, nm);

    timer.stopTimer(datasetname + ".time");

    CommonFunctions::writeGramMatrixToFile(nm, data, datasetname);

}


void ExplicitLineGraphKernel::getS(TemporalGraphStream &tgs, vector<NodeId> &S) {
    for (unsigned int i = 0; i < tgs.get_num_nodes(); ++i) {
        S.push_back(i);
    }
}


void ExplicitLineGraphKernel::randomWalkVertexEdgeLabelRek(std::vector<std::vector<Label>> &seqs, std::vector<Label> seq, TemporalGraph &tg, NodeId s, unsigned int k, Time curtime) {
    if (k == 0) {
        return;
    }

    for (TemporalEdge &e : tg.nodes.at(s).adjlist) {
        vector<Label> nseq = seq;
        Label l = 10 * (1 + tg.nodes.at(s).getLabel(e.t)) + (1 + tg.nodes.at(e.v_id).getLabel(e.t+1));
        nseq.push_back(l);

        if (e.t < curtime) continue;

        if (e.t == curtime || seq.size() == 1 || curtime == 0)
            nseq.push_back(8);
        else
            nseq.push_back(9);

        seqs.push_back(nseq);

        randomWalkVertexEdgeLabelRek(seqs, nseq, tg, e.v_id, k-1, e.t + 1);
    }
}


vector<vector<Label>> ExplicitLineGraphKernel::temporalRandomWalkVertexEdgeLabel(TemporalGraph &tg, NodeId s, unsigned int k) {

    vector<vector<Label>> seqs;
    std::vector<Label> seq;

    randomWalkVertexEdgeLabelRek(seqs, seq, tg, s, k, 0);

    return seqs;
}


vector<vector<Label>> ExplicitLineGraphKernel::temporalRandomWalkVertexEdgeLabelIt(TemporalGraph &tg, NodeId s, unsigned int k) {

    vector<vector<Label>> seqs;

    struct ns {
        NodeId n;
        std::vector<Label> seq;
        Time t;
        unsigned int k;
    };

    vector<ns> q;
    q.push_back(ns{s, std::vector<Label>(), 0, 0});

    while (!q.empty()) {
        ns cur = q.back();
        q.pop_back();
        if (cur.seq.size() > 0) {
//            cout << seqToString(cur.seq) << endl;
            seqs.push_back(cur.seq);
        }

        if (cur.k == k) continue;

        for (TemporalEdge &e : tg.nodes.at(cur.n).adjlist) {

            if (e.t < cur.t) continue;
            vector<Label> seq = cur.seq;

            Label cur_label = 10 * (1 + tg.nodes.at(cur.n).getLabel(e.t)) + (1 + tg.nodes.at(e.v_id).getLabel(e.t+1));
            seq.push_back(cur_label);

            if (cur.t == e.t || cur.t == 0)
                seq.push_back(9);
            else
                seq.push_back(8);

            ns nsnew{e.v_id, seq, e.t+1, cur.k + 1};
            q.push_back(nsnew);
        }
    }

    return seqs;
}
