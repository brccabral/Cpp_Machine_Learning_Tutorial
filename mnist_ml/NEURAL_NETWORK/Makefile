CC=g++
SRC := $(MNIST_ML_ROOT)/NEURAL_NETWORK/src
CFLAGS := -std=c++11 -DEUCLID -g
INCLUDE_DIR := $(MNIST_ML_ROOT)/NEURAL_NETWORK/include

all: main

main: obj $(SRC)/network.cc obj/layer.o obj/neuron.o
	$(CC) $(CFLAGS) $(SRC)/network.cc -o main -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -L$(MNIST_ML_ROOT)/lib -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/layer.o: $(SRC)/layer.cc
	$(CC) -c $(CFLAGS) $(SRC)/layer.cc -o obj/layer.o -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include

obj/neuron.o: $(SRC)/neuron.cc
	$(CC) -c $(CFLAGS) $(SRC)/neuron.cc -o obj/neuron.o -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include

clean:
	-rm -r main obj