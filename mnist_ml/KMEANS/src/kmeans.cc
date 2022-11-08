#include "kmeans.hpp"

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indexes = new std::unordered_set<int>;
}

void kmeans::init_clusters()
{
    // create cluster for each label/class
    for (int i = 0; i < num_clusters; i++)
    {
        int index = (rand() % training_data->size()); // number between [0 and train.size-1]
        // loop until find an unused label
        while (used_indexes->find(index) != used_indexes->end())
        {
            index = (rand() % training_data->size());
        }
        clusters->push_back(new cluster(training_data->at(index))); // create cluster with first point found for that label
        used_indexes->insert(index); // set label as found
    }
}

void kmeans::init_clusters_for_each_class() {}

void kmeans::train() {}

double kmeans::euclidean_distance(std::vector<double> *, data *) {}

double kmeans::validate() {}

double kmeans::test() {}