/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "TemporalGraphKernelApprox.h"
#include "GraphKernelCommonFunctions.h"
#include <unordered_map>
#include <Eigen/Sparse>
#include <iterator>
#include "Timer.h"

using namespace std;

void TemporalGraphKernelApprox::calculaleGramMatrix (TemporalGraphStreams &data, string datasetname,
                                                     unsigned int const kmin, unsigned int const kmax, unsigned int S_size) {
    for (unsigned int h = kmin; h <= kmax; ++h) {
        string fdatasetname = datasetname + to_string(h) +".gram";
        getGramMatrix(data, fdatasetname, h, S_size);
    }
}

string TemporalGraphKernelApprox::seqToString(vector<Label> seq) {
    std::stringstream result;
    std::copy(seq.begin(), seq.end(), std::ostream_iterator<Label>(result, " "));
    return result.str();
}


vector<string> TemporalGraphKernelApprox::splitseq(vector<Label> seq) {
    vector<string> results;
    for (unsigned long  i = 1; i < seq.size(); i += 2) {
        vector<Label> seq0;
        unsigned long j = 0;
        for (j = 0; j < i; ++j) {
            seq0.push_back(seq.at(j));
        }
        seq0.push_back(seq.at(j));
        string r0 = seqToString(seq0);
        results.push_back(r0);
    }
    return results;
}

void TemporalGraphKernelApprox::getGramMatrix(TemporalGraphStreams &data, std::string datasetname, unsigned int k,
                                              unsigned int S_size) {

    Timer timer;

    using Histogram = unordered_map<string, unsigned int>;
    using S = Eigen::Triplet<unsigned int>;

    vector<Histogram> histograms;
    unsigned int numSeqs = 0;
    map<string, unsigned int> sequenceNumbering;

    timer.startTimer();
    for (TemporalGraphStream &tgs : data) {
        Histogram histogram;
        vector<NodeId> S;
        getS(tgs, S_size, S);

        for (NodeId s : S) {
            vector<Label> seq;
            seq = temporalRandomWalkVertexLabel(tgs, s, k);
            vector<string> seqstrs = splitseq(seq);
            for (string seqstr : seqstrs) {
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

std::vector<Label> TemporalGraphKernelApprox::temporalRandomWalkVertexLabel(TemporalGraphStream &tgs, NodeId s, unsigned int k) {
    TemporalGraph tg = tgs.toTemporalGraph();
    std::vector<Label> seq;

    NodeId cur = s;
    Time curTime = 0;
    for (size_t i = 0; i < k; ++i) {

        TemporalEdges tes;
        for (TemporalEdge &e : tg.nodes.at(cur).adjlist) {
            if (e.t >= curTime) {
                tes.push_back(e);
            }
        }

        if (tes.empty()) {
            return seq;
        }

        unsigned long next = rand() % tes.size();
        TemporalEdge &e = tes.at(next);

        Label cur_label = 10 * (1 + tg.nodes.at(cur).getLabel(e.t)) + (1 + tg.nodes.at(e.v_id).getLabel(e.t+1));
        seq.push_back(cur_label);

        if (curTime == e.t || curTime == 0)
            seq.push_back(9);
        else
            seq.push_back(8);
        cur = e.v_id;
        curTime = e.t + 1;
    }

    return seq;
}

void TemporalGraphKernelApprox::getS(TemporalGraphStream &tgs, unsigned int S_size, vector<NodeId> &S) {
    for (size_t i = 0; i < S_size; ++i) {
        S.push_back(rand() % tgs.get_num_nodes());
    }
}
