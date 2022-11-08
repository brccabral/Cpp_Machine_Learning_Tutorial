#include "kmeans.hpp"

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indexes = new std::unordered_set<int>;
}

// create K number of clusters
void kmeans::init_clusters()
{
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

// create cluster for each label/class
void kmeans::init_clusters_for_each_class()
{
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

void kmeans::train()
{
    while (used_indexes->size() < training_data->size())
    {
        // pick a random point
        int index = (rand() % training_data->size());
        // make sure we are not picking the same point
        while (used_indexes->find(index) != used_indexes->end())
        {
            index = (rand() % training_data->size());
        }
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        // find nearest centroid from training point
        for (int j = 0; j < clusters->size(); j++)
        {
            double current_dist = euclidean_distance(clusters->at(j)->centroid, training_data->at(index));
            if (current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        // add point to nearest cluster
        clusters->at(best_cluster)->add_to_cluster(training_data->at(index));
        used_indexes->insert(index);
    }
}

double kmeans::euclidean_distance(std::vector<double> *, data *) {}

double kmeans::validate() {}

double kmeans::test() {}