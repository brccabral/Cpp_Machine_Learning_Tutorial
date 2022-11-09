#pragma once

#include "neuron.hpp"
#include <vector>

class Layer
{
public:
    int currentLayerSize;
    std::vector<Neuron *> neurons;
    std::vector<double> layerOutput;
    Layer(int, int);
};