#include "data_handler.hpp"

data_handler::data_handler()
{
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    test_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}
data_handler::~data_handler() {}

void data_handler::read_feature_vector(std::string path) {}
void data_handler::read_feature_labels(std::string path) {}
void data_handler::split_data() {}
void data_handler::count_classes() {}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes) {}

std::vector<data *> *data_handler::get_training_data() {}
std::vector<data *> *data_handler::get_test_data() {}
std::vector<data *> *data_handler::get_validation_data() {}
