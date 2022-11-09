#pragma once

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "input_layer.hpp"
#include "hidden_layer.hpp"
#include "output_layer.hpp"
#include "common.hpp"

class Network : public common_data
{
private:
    InputLayer *inputLayer;
    OutputLayer *outputLayer;
    std::vector<HiddenLayer *> hiddenLayers;
    double eta; // learning rate

public:
    // number of neurons for each hidden layer, neurons input, neurons output
    Network(std::vector<int> hiddenLayerSpec, int, int);
    ~Network();
    void fprop(data *data); // forward
    void bprop(data *data); // backward
    void updateWeights();
    void train();
    void validate();
    void test();
};