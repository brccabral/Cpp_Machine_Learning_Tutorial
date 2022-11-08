#include "kmeans.hpp"

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indexes = new std::unordered_set<int>;
}

void kmeans::init_clusters()
{
    // create K number of clusters
    for (int i = 0; i < num_clusters; i++)
    {
        // pick a random point to be the centroid
        int index = (rand() % training_data->size()); // number between [0 and train.size-1]
        // make sure we are not picking the same point
        while (used_indexes->find(index) != used_indexes->end())
        {
            index = (rand() % training_data->size());
        }
        clusters->push_back(new cluster(training_data->at(index))); // create cluster with a random new point
        used_indexes->insert(index);                                // save selected point
    }
}

void kmeans::init_clusters_for_each_class()
{
    // create cluster for each label/class
    std::unordered_set<int> classes_used;
    for (int i = 0; i < training_data->size(); i++)
    {
        // if new class is found, add to the list
        if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end())
        {
            clusters->push_back(new cluster_t(training_data->at(i)));
            classes_used.insert(training_data->at(i)->get_label());
            used_indexes->insert(i);
        }
    }
}

void kmeans::train() {}

double kmeans::euclidean_distance(std::vector<double> *, data *) {}

double kmeans::validate() {}

double kmeans::test() {}