/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "ExplicitRW.h"
#include "GraphKernelCommonFunctions.h"
#include <unordered_map>
#include <Eigen/Sparse>
#include <iterator>
#include <queue>
#include "Timer.h"

using namespace std;

void ExplicitRW::calculaleGramMatrix (TemporalGraphStreams &data, string datasetname,
                                                     unsigned int const kmin, unsigned int const kmax) {
    for (unsigned int h = kmin; h <= kmax; ++h) {
        string fdatasetname = datasetname + to_string(h) +".gram";
        getGramMatrix(data, fdatasetname, h);
    }
}

string ExplicitRW::seqToString(vector<Label> seq) {
    std::stringstream result;
    std::copy(seq.begin(), seq.end(), std::ostream_iterator<Label>(result, " "));
    return result.str();
}

void ExplicitRW::getGramMatrix(TemporalGraphStreams &data, std::string datasetname, unsigned int k) {

    Timer timer;

    using Histogram = unordered_map<string, unsigned int>;
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
            seqs = randomWalkVertexEdgeLabelIt(tg, s, k);

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


void ExplicitRW::getS(TemporalGraphStream &tgs, vector<NodeId> &S) {
    for (size_t i = 0; i < tgs.get_num_nodes(); ++i) {
        S.push_back(i);
    }
}


void ExplicitRW::randomWalkVertexEdgeLabelRek(std::vector<std::vector<Label>> &seqs, std::vector<Label> seq, TemporalGraph &tg, NodeId &s, unsigned int k) {
    seq.push_back(tg.nodes.at(s).static_label);
    seqs.push_back(seq);
    if (k == 0) {
        return;
    }
    for (TemporalEdge &e : tg.nodes.at(s).adjlist) {
        vector<Label> nseq = seq;
        nseq.push_back(e.t);
        randomWalkVertexEdgeLabelRek(seqs, nseq, tg, e.v_id, k-1);
    }
}

vector<vector<Label>> ExplicitRW::randomWalkVertexEdgeLabel(TemporalGraph &tg, NodeId s, unsigned int k) {
    vector<vector<Label>> seqs;
    std::vector<Label> seq;

    randomWalkVertexEdgeLabelRek(seqs, seq, tg, s, k);

    return seqs;
}


vector<vector<Label>> ExplicitRW::randomWalkVertexEdgeLabelIt(TemporalGraph &tg, NodeId s, unsigned int k) {

    vector<vector<Label>> seqs;

    struct ns {
        NodeId n;
        std::vector<Label> seq;
        unsigned int k;
    };

    queue<ns> q;
    q.emplace(ns{s, std::vector<Label>(), 0});

    while (!q.empty()) {
        ns cur = q.front();
        q.pop();

        cur.seq.push_back(tg.nodes.at(cur.n).static_label);

//        if (cur.seq.size() > 1) {
            seqs.push_back(cur.seq);
//            cout << seqToString(cur.seq) << endl;
//        }

        if (cur.k == k) continue;

        for (TemporalEdge &e : tg.nodes.at(cur.n).adjlist) {

            vector<Label> seq = cur.seq;
            seq.push_back(e.t);

            ns next_ns{e.v_id, seq, cur.k + 1};
            q.emplace(next_ns);
        }
    }

    return seqs;
}

