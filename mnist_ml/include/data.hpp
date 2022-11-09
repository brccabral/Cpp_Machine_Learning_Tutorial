#pragma once

#include <vector>
#include "stdint.h"
#include "stdio.h"

class data
{
    std::vector<uint8_t> *feature_vector; // No class/label at end.
    std::vector<double> *double_feature_vector;
    std::vector<int> *class_vector;
    std::vector<double> *normalizedFeatureVector;
    uint8_t label;
    int enum_label; // A -> 1, B -> 2
    double distance;

public:
    data();
    ~data();
    void set_feature_vector(std::vector<uint8_t> *);
    void append_to_feature_vector(uint8_t);

    void set_feature_vector(std::vector<double> *);
    void append_to_feature_vector(double);
    void set_class_vector(int count);

    void set_label(uint8_t);
    void set_enumerated_label(int);
    void set_distance(double);
    double get_distance();

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enumerated_label();

    std::vector<uint8_t> *get_feature_vector();
    std::vector<double> *get_double_feature_vector();
    std::vector<int> *get_class_vector();

    void set_normalized_featureVector(std::vector<double> *vect);
    std::vector<double> *get_normalized_featureVector();
};