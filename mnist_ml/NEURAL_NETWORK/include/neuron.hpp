#pragma once

#include <vector>
#include <cmath>

class Neuron
{
    std::vector<double> weights;
    double preActivation;
    double activatedOutput;
    double outputDerivative;
    double error;
    double alpha; // used in activation functions

public:
    Neuron(int, int);
    ~Neuron();
    void initializeWeights(int previousLayerSize, int currentLayerSize);
    void setError(double);
    void setWeight(double, int);

    double calculatePreActivation(std::vector<double>);
    double calculateOutputDerivate();
    double sigmoid();
    double relu();
    double leakyRelu();
    double inverseSqrtRelu();

    double getOutput();
    double getOutputDerivative();
    double getError();
    std::vector<double> getWeights();
};