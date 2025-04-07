# Evaluation_Task
evaluation task to be reviewed by the compiler research team
## Repository Structure

### Test Cases (`tests/`)
- [`test_basicforwardpass.cpp`](tests/test_basicforwardpass.cpp): Basic examples of computing gradients and Jacobians
- [`test_branching.cpp`](tests/test_branching.cpp): Demonstrates AD with ReLU branching operations
- [`test_discontinuity.cpp`](tests/test_discontinuity.cpp): Shows handling of ReLU discontinuities
- [`test_gradient.cpp`](tests/test_gradient.cpp): Simple gradient computation example
- [`test_hidden.cpp`](tests/test_hidden.cpp): Implementation of hidden layer computations

### LLM Entity Examples (`clad_examples/`)
- [`train_multiple_neurons.cpp`](clad_examples/train_multiple_neurons.cpp): Training simple entity with clad integration
- [`train_transformer.cpp`](clad_examples/train_transformer.cpp): Simple transformer model implementation

to run `train_transformer.cpp`:

```sudo apt-get install libeigen3-dev
clang++ -O3 -std=c++20 -fopenmp train_transformer.cpp -o train_transformer
./train_transformer
```

to run `train_multiple_neurons.cpp` in clad environment:

```clang++ -std=c++17 -O0 -g \
-I/root/llvm-project/llvm/tools/clad/include \
-I/usr/include/eigen3 \
-fplugin=/root/llvm-project/llvm/tools/clad/build/lib/clad.so \
train_multiple_neurons.cpp -o multiple
./multiple```
