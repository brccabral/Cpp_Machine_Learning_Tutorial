#pragma once

#include <vector>
#include <cmath>
#include <stdio.h>

class Neuron
{
public:
    double output;
    double delta;
    std::vector<double> weights;
    Neuron(int, int);
    void initializeWeights(int);
};