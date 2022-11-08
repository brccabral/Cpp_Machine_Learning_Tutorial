CC=g++
MNIST_ML_ROOT := $(if $(MNIST_ML_ROOT),$(MNIST_ML_ROOT),$(PWD)/..)
SRC := $(MNIST_ML_ROOT)/KNN/src
CFLAGS := -std=c++11 -DEUCLID
INCLUDE_DIR := $(MNIST_ML_ROOT)/KNN/include

all: main

main: $(SRC)/knn.cc
	$(CC) $(CFLAGS) $(SRC)/knn.cc -o main -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -L$(MNIST_ML_ROOT)/lib -ldata

clean:
	rm main