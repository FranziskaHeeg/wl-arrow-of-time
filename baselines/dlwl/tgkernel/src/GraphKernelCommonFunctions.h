/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_ABSTRACTKERNEL_H
#define TGKERNEL_ABSTRACTKERNEL_H

#include "TemporalGraphs.h"
#include <Eigen/Core>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iostream>

using MatrixXl = Eigen::Matrix<unsigned long, Eigen::Dynamic, Eigen::Dynamic >;

namespace CommonFunctions {

    template<typename T, typename U>
    void normalizeGramMatrix(T &m, U &nm) {
        unsigned long size = m.rows();
        nm.resize(size, size);
        for (unsigned int i = 0; i < size; ++i) {
            for (unsigned int j = 0; j < size; ++j) {
                double x = (sqrt(m(i, i)) * sqrt(m(j, j)));
                if (x != 0)
                    nm(i, j) = m(i, j) / x;
                else
                    nm(i, j) = 0;
                if (std::isnan(nm(i, j))) nm(i, j) = 0;
            }
        }
    }

    // write libsvm style gram matrix to file
    template<typename T>
    void writeGramMatrixToFile(T m, TemporalGraphStreams &data, std::string const &filename) {
        std::ofstream gramFile;
        gramFile.open(filename);
        for (size_t i = 0; i < data.size(); ++i) {
            gramFile << data.at(i).label << " 0:" << (i + 1);
            for (size_t c = 0; c < data.size(); ++c) {
                gramFile << " " << (c + 1) << ":" << m(i, c);
            }
            gramFile << std::endl;
        }
        gramFile.close();
    }

    // map each product (v_1, v_2) of vertics to a number H(v_1, v_2)
    template<typename T>
    unsigned int productMapping(std::vector<Label> &v1_label, std::vector<Label> &v2_label, T &H) {
        unsigned int n_vx = 0;
        for (unsigned int i = 0; i < v1_label.size(); ++i) {
            for (unsigned int j = 0; j < v2_label.size(); ++j) {
                if (v1_label[i] == v2_label[j]) {
                    H(i, j) = n_vx;
                    n_vx++;
                }
            }
        }
        return n_vx;
    }

}
#endif //TGKERNEL_ABSTRACTKERNEL_H
