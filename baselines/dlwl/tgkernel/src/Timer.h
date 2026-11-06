/**
This file is part of TGKernel.
Copyright (C) 2020 Lutz Oettershagen
Contact: lutz.oettershagen@cs.uni-bonn.de
**/

#ifndef TGKERNEL_TIMER_H
#define TGKERNEL_TIMER_H

#include <chrono>
#include <iostream>


class Timeout {};

class Timer {

public:

    void writeOutTime(std::chrono::time_point<std::chrono::high_resolution_clock> t1,
                      std::chrono::time_point<std::chrono::high_resolution_clock> t2, std::string fname);
    void startTimer();
    void stopTimer(std::string fname);

    void checkTimeout();

private:

    unsigned int const TIMEOUT_MINS = 60 * 24;

    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    std::chrono::time_point<std::chrono::high_resolution_clock> t2;
};


#endif //TGKERNEL_TIMER_H
