#include "kmeans.hpp"

kmeans::kmeans(int k) {
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indexes = new std::unordered_set<int>;
}

void kmeans::init_clusters() {}

void kmeans::init_clusters_for_each_class() {}

void kmeans::train() {}

double kmeans::euclidean_distance(std::vector<double> *, data *) {}

double kmeans::validate() {}

double kmeans::test() {}