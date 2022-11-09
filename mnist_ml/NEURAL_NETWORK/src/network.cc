#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate)
{
    for (int i = 0; i < spec.size(); i++)
    {
        if (i == 0) // first hidden layer
            layers.push_back(new Layer(inputSize, spec.at(i)));
        else
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i)));
    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), numClasses));
    this->learningRate = learningRate;
}

Network::~Network() {}

double Network::activate(std::vector<double> weights, std::vector<double> input)
{
    double activation = weights.back();          // bias term
    for (int i = 0; i < weights.size() - 1; i++) // -1 is to ignore bias
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

// sigmoid
double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

// sigmoid derivative
double Network::transferDerivative(double output)
{
    return output * (1 - output);
}