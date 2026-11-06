/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#include "Timer.h"
#include <fstream>



void Timer::writeOutTime(std::chrono::time_point<std::chrono::high_resolution_clock> t1,
                  std::chrono::time_point<std::chrono::high_resolution_clock> t2, std::string fname) {
    std::ofstream time_file;
    time_file.open(fname);
    time_file << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
    time_file << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << " seconds\n";
    time_file << std::chrono::duration_cast<std::chrono::minutes>(t2 - t1).count() << " minutes\n";
    time_file << std::chrono::duration_cast<std::chrono::hours>(t2 - t1).count() << " hours\n";
    time_file.close();
}

void Timer::startTimer() {
    t1 = std::chrono::high_resolution_clock::now();
}

void Timer::stopTimer(std::string fname) {
    t2 = std::chrono::high_resolution_clock::now();
    writeOutTime(t1, t2, fname);
}

void Timer::checkTimeout() {
    t2 = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::minutes>(t2 - t1).count() > TIMEOUT_MINS)
        throw Timeout();
}