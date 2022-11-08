#pragma once

#include "layer.hpp"
#include "data.hpp"

class OutputLayer : public Layer
{
public:
    OutputLayer(int prev, int current) : Layer(prev, current) {}

    void feedForward(Layer prev);
    void backProp(data *data);
    void updateWeights(double, Layer *);
};