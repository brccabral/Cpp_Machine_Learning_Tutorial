#pragma once

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

class data_handler
{
    std::vector<data *> *data_array; // all of the data (pre-split)
    std::vector<data *> *training_data;
    std::vector<data *> *test_data;
    std::vector<data *> *validation_data;

    int class_counts;
    int feature_vector_size;
    std::map<uint8_t, int> class_map;
    std::map<std::string, int> classMap;

    double TRAIN_SET_PERCENT = 0.1;
    double TEST_SET_PERCENT = 0.075;
    double VALIDATION_PERCENT = 0.005;
    // const double TRAIN_SET_PERCENT = 0.75;
    // const double TEST_SET_PERCENT = 0.20;
    // const double VALIDATION_PERCENT = 0.05;

public:
    data_handler();
    ~data_handler();
    void set_train_percent(double perc) { TRAIN_SET_PERCENT = perc; };
    void set_test_percent(double perc) { TEST_SET_PERCENT = perc; };
    void set_validation_percent(double perc) { VALIDATION_PERCENT = perc; };

    void read_feature_vector(std::string path);
    void read_feature_labels(std::string path);
    void read_input_data(std::string path);
    void read_label_data(std::string path);
    void split_data();
    void count_classes();
    void read_csv(std::string path, std::string delimiter);

    uint32_t convert_to_little_endian(const unsigned char *bytes);
    uint32_t format(const unsigned char *bytes);

    std::vector<data *> *get_training_data();
    std::vector<data *> *get_test_data();
    std::vector<data *> *get_validation_data();

    int get_class_count();
    int get_data_array_size();
    int get_training_data_size();
    int get_test_data_size();
    int get_validation_data_size();
    void print();
    void normalize();
};