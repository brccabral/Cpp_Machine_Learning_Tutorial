#include "data.hpp"

data::data()
{
    feature_vector = new std::vector<uint8_t>;
}

void data::set_feature_vector(std::vector<uint8_t> *vect)
{
    feature_vector = vect;
}

void data::append_to_feature_vector(uint8_t val)
{
    feature_vector->push_back(val);
}

void data::append_to_feature_vector(double val)
{
    normalizedFeatureVector->push_back(val);
}

void data::set_class_vector(int count)
{
    class_vector = new std::vector<int>();
    for (int i = 0; i < count; i++)
    {
        if (i == label)
        {
            class_vector->push_back(1);
        }
        else
        {
            class_vector->push_back(0);
        }
    }
}

void data::set_label(uint8_t val)
{
    label = val;
}

void data::set_enumerated_label(int val)
{
    enum_label = val;
}

void data::set_distance(double val)
{
    distance = val;
}

double data::get_distance()
{
    return distance;
}

int data::get_feature_vector_size()
{
    return feature_vector->size();
}

uint8_t data::get_label()
{
    return label;
}

uint8_t data::get_enumerated_label()
{
    return enum_label;
}

std::vector<uint8_t> *data::get_feature_vector()
{
    return feature_vector;
}

void data::set_normalized_featureVector(std::vector<double> *vect)
{
    normalizedFeatureVector = vect;
}

std::vector<double> *data::get_normalized_featureVector()
{
    return normalizedFeatureVector;
}

std::vector<int> data::get_class_vector()
{
    return *class_vector;
}