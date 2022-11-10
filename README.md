# C++ Machine Learning Tutorial

Tutorial from Gerard Taylor https://www.youtube.com/playlist?list=PL79n_WS-sPHKklEvOLiM1K94oJBsGnz71

MNIST data can be downloaded from http://yann.lecun.com/exdb/mnist/  
Iris Data Set can be downloaded from https://archive.ics.uci.edu/ml/datasets/iris  

- Compile `libdata.so`
```console
cd mnist_ml
export MNIST_ML_ROOT=$PWD
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/lib
```

- Then, for each ML algorithm, `make` and execute `main`
```console
cd KMEANS # or KNN or NEURAL_NETWORK
make clean && make
./main
```

NEURAL_NETWORK can use MNIST data or Iris data. To use Iris, remove `-DMNIST` from its Makefile.