#pragma once

#include "data.hpp"
#include "neuron.hpp"
#include "layer.hpp"
#include "common.hpp"

class Network : public common_data
{
public:
    std::vector<Layer *> layers;
    double learningRate;
    double testPerformance;
    // number of neurons for each hidden layer, neurons input, neurons output
    Network(std::vector<int> hiddenLayerSpec, int, int, double);
    ~Network();
    std::vector<double> fprop(data *data); // forward, returns last layer
    void bprop(data *data);                // backward
    void updateWeights(data *data);
    void train(int); // num iterations
    void validate();
    double test();
    double activate(std::vector<double>, std::vector<double>); // dot product (weight*input)
    double transfer(double);                                   // sigmoid
    double transferDerivative(double);                         // used for backprop
    int predict(data *data);                                   // return the index of the maximum value in the output array.
};