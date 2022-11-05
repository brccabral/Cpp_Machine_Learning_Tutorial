#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"

knn::knn(int val)
{
    k = val;
}
knn::knn() {}
knn::~knn() {}

void knn::set_training_data(std::vector<data *> *vect)
{
    training_data = vect;
}
void knn::set_test_data(std::vector<data *> *vect)
{
    test_data = vect;
}
void knn::set_validation_data(std::vector<data *> *vect)
{
    validation_data = vect;
}

void knn::find_knearest(data *query_point) {}
void knn::set_k(int val) {}

int knn::predict() {}
double knn::calculate_distance(data *query_point, data *input)
{
    double distance = 0.0;
    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        printf("ERROR: vector size and input size mismatch\n");
        exit(1);
    }

#ifdef EUCLID
    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
#endif
    return distance;
}
double knn::validate_performance() {}
double knn::test_performance() {}