#include "data_handler.hpp"
#include <algorithm>
#include <random>

data_handler::data_handler()
{
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    test_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}

data_handler::~data_handler() {}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // |MAGIC|NUM IMAGES|ROWSIZE|COLSIZE|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 4; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Input file header.\n");
        int image_size = header[2] * header[2];
        for (int i = 0; i < header[1]; i++)
        {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++)
            {
                if (fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    printf("Error reading from file.\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
    }
    else
    {
        printf("Could not find file\n");
        exit(1);
    }
}

void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // |MAGIC|NUM IMAGES|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Label file header.\n");
        for (int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                printf("Error reading from file.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored label.\n");
    }
    else
    {
        printf("Could not find file\n");
        exit(1);
    }
}

void data_handler::read_input_data(std::string path)
{
    uint32_t magic = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        int i = 0;
        while (i < 4)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                switch (i)
                {
                case 0:
                    magic = convert_to_little_endian(bytes);
                    i++;
                    break;
                case 1:
                    num_images = convert_to_little_endian(bytes);
                    i++;
                    break;
                case 2:
                    num_rows = convert_to_little_endian(bytes);
                    i++;
                    break;
                case 3:
                    num_cols = convert_to_little_endian(bytes);
                    i++;
                    break;
                }
            }
        }
        printf("Done getting file header.\n");
        uint32_t image_size = num_rows * num_cols;
        for (i = 0; i < num_images; i++)
        {
            data *d = new data();
            d->set_feature_vector(new std::vector<uint8_t>());
            uint8_t element[1];
            for (int j = 0; j < image_size; j++)
            {
                if (fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
            }
            data_array->push_back(d);
            data_array->back()->set_class_vector(class_counts);
        }
        normalize();
        feature_vector_size = data_array->at(0)->get_feature_vector()->size();
        printf("Successfully read %lu data entries.\n", data_array->size());
        printf("The Feature Vector Size is: %d\n", feature_vector_size);
    }
    else
    {
        printf("Invalid Input File Path\n");
        exit(1);
    }
}

void data_handler::read_label_data(std::string path)
{
    uint32_t magic = 0;
    uint32_t num_images = 0;
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        int i = 0;
        while (i < 2)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                switch (i)
                {
                case 0:
                    magic = convert_to_little_endian(bytes);
                    i++;
                    break;
                case 1:
                    num_images = convert_to_little_endian(bytes);
                    i++;
                    break;
                }
            }
        }

        for (unsigned j = 0; j < num_images; j++)
        {
            uint8_t element[1];
            if (fread(element, sizeof(element), 1, f))
            {
                data_array->at(j)->set_label(element[0]);
            }
        }

        printf("Done getting Label header.\n");
    }
    else
    {
        printf("Invalid Label File Path\n");
        exit(1);
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int valid_size = data_array->size() * VALIDATION_PERCENT;

    std::random_shuffle(data_array->begin(), data_array->end());

    // Training Data

    int count = 0;
    int index = 0;
    while (count < train_size)
    {
        training_data->push_back(data_array->at(index++));
        count++;
    }

    // Test Data
    count = 0;
    while (count < test_size)
    {
        test_data->push_back(data_array->at(index++));
        count++;
    }

    // Test Data

    count = 0;
    while (count < valid_size)
    {
        validation_data->push_back(data_array->at(index++));
        count++;
    }

    printf("Training data size: %lu.\n", training_data->size());
    printf("Test data size: %lu.\n", test_data->size());
    printf("Validation data size: %lu.\n", validation_data->size());
}

void data_handler::count_classes()
{
    int count = 0;
    for (unsigned int i = 0; i < data_array->size(); i++)
    {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            count++;
        }
    }
    class_counts = count;
    for (data *data : *data_array)
        data->set_class_vector(class_counts);
    printf("Successfully extracted %d unique classes\n", class_counts);
}

void data_handler::normalize()
{
    std::vector<double> mins, maxs;

    data *d = data_array->at(0);
    for (auto val : *d->get_feature_vector())
    {
        mins.push_back(val);
        maxs.push_back(val);
    }

    for (int i = 1; i < data_array->size(); i++)
    {
        d = data_array->at(i);
        for (int j = 0; j < d->get_feature_vector_size(); j++)
        {
            double value = (double)d->get_feature_vector()->at(j);
            if (value < mins.at(j))
                mins[j] = value;
            if (value > maxs.at(j))
                maxs[j] = value;
        }
    }

    for (int i = 0; i < data_array->size(); i++)
    {
        data_array->at(i)->set_normalized_featureVector(new std::vector<double>());
        data_array->at(i)->set_class_vector(class_counts);
        for (int j = 0; j < data_array->at(i)->get_feature_vector_size(); j++)
        {
            if (maxs[j] - mins[j] == 0)
                data_array->at(i)->append_to_feature_vector(0.0);
            else
                data_array->at(i)->append_to_feature_vector(
                    (double)(data_array->at(i)->get_feature_vector()->at(j) - mins[j]) / (maxs[j] - mins[j]));
        }
    }
}

void data_handler::read_csv(std::string path, std::string delimiter)
{
    class_counts = 0;
    std::ifstream data_file(path.c_str());
    std::string line; // holds each line
    while (std::getline(data_file, line))
    {
        if (line.length() == 0)
            continue;

        data *d = new data();
        d->set_normalized_featureVector(new std::vector<double>());
        size_t position = 0;
        std::string token; // value in between delimiter
        while ((position = line.find(delimiter)) != std::string::npos)
        {
            token = line.substr(0, position);
            d->append_to_feature_vector(std::stod(token));
            line.erase(0, position + delimiter.length());
        }
        if (classMap.find(line) != classMap.end())
        {
            d->set_label(classMap[line]);
        }
        else
        {
            classMap[line] = class_counts;
            d->set_label(classMap[line]);
            class_counts++;
        }
        data_array->push_back(d);
    }
    for (data *data : *data_array)
        data->set_class_vector(class_counts);

    feature_vector_size = data_array->at(0)->get_normalized_featureVector()->size();
}

uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)
{
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

std::vector<data *> *data_handler::get_training_data()
{
    return training_data;
}

std::vector<data *> *data_handler::get_test_data()
{
    return test_data;
}

std::vector<data *> *data_handler::get_validation_data()
{
    return validation_data;
}

int data_handler::get_class_count()
{
    return class_counts;
}

int data_handler::get_data_array_size()
{
    return data_array->size();
}

int data_handler::get_training_data_size()
{
    return training_data->size();
}

int data_handler::get_test_data_size()
{
    return test_data->size();
}

int data_handler::get_validation_data_size()
{
    return validation_data->size();
}

void data_handler::print()
{
    printf("Training Data:\n");
    for (auto data : *training_data)
    {
        for (auto value : *data->get_normalized_featureVector())
        {
            printf("%.3f,", value);
        }
        printf(" ->   %d\n", data->get_label());
    }

    printf("Test Data:\n");
    for (auto data : *test_data)
    {
        for (auto value : *data->get_normalized_featureVector())
        {
            printf("%.3f,", value);
        }
        printf(" ->   %d\n", data->get_label());
    }

    printf("Validation Data:\n");
    for (auto data : *validation_data)
    {
        for (auto value : *data->get_normalized_featureVector())
        {
            printf("%.3f,", value);
        }
        printf(" ->   %d\n", data->get_label());
    }
}

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../MNIST/train-images.idx3-ubyte"); // for now, needs to come first
    dh->read_feature_labels("../MNIST/train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}