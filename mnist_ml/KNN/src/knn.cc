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
double knn::calculate_distance(data *query_point, data *input) {}
double knn::validate_performance() {}
double knn::test_performance() {}