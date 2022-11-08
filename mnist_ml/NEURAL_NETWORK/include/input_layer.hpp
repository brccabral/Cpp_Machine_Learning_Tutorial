#pragma once

#include "layer.hpp"
#include "data.hpp"

class InputLayer : public Layer
{
public:
    InputLayer(int prev, int current) : Layer(prev, current) {}

    void setLayerOutputs(data *d);
};