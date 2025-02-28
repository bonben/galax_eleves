# Galax : Simulate the collision of two galaxies

## Description

Galax is the simple implementation of the [n-body problem](https://en.wikipedia.org/wiki/N-body_problem), applied to the simulation of celestial objects. The equations that describe the problem between are listed in [equations.pdf](equations.pdf).

## Code

A naive implementation can be found in [Model_CPU_naive.cpp](src/Model/Model_CPU/Model_CPU_naive/Model_CPU_naive.cpp). Note that the implementation uses "magic" numbers at some stages. We do not want to focus on the accuracy of the simulation but on the parallelisation of the nbody problem.

You can generate the hierarchy of the different classes as well as a list of the different attributes and methods used in the project thanks to the doxygen tool. The instructions to perform this operation are detailed in the [doc](doc) directory.

## Means

You will work in groups of 2 (except one group of three). Fill the document here : [Groups](https://partage.imt.fr/index.php/s/NMj4Ag9ewyxpCen).

Please specify if you plan to implement a GPU version of galax in the corresponding column.

## Objective

The objective of the course is to accelerate the simulation. You must maximize the number of simulated frames per second, when the display is disabled and the number of simulated particles is 10000. You can use all the techniques that have been seen in the course, and combine them.

* 2 categories:
  * Exact reproduction of [Model_CPU_naive.cpp](src/Model/Model_CPU/Model_CPU_naive/Model_CPU_naive.cpp)
  * Modifications of the algorithm allowed

**The best group will be given access to a massively parallel machine to test its code (2xAMD EPYC 7643 48-Core Processor (192 threads) + 4xA100 GPUs).**

## Evaluation

* Project logbook: you will keep an up-to-date document explaining your experiences, your results, your choices, taken from day to day. No need to detail, to put images, to spend time on formatting. You will use text format files that have been created for you [here](https://partage.imt.fr/index.php/s/JNRsp7tqkz3RiFT).
* Presentation (10 minutes + 5 minutes questions)
  * Steps fo final implementation
  * Summary of all results for each step
  * Critical analysis of the measures
  * Demonstration
* Deliverables
  * Project code

### Extras

Any extra feature will be well appreciated and taken into account for the evaluation :
* Improvement of the graphical rendering
* Inclusion of other libraries into the cmake build automation
* Addition of algorithmic variants of the nbody problem (n² -> nlogn)
* Include Galax as a wrapper for other languages
* ...
## How to compile and run the code

Galax uses `cmake` in order to automate building.
The following procedure has been tested on a fresh Ubuntu 20.04, but it can be adapted for other platforms.

### First steps

Galax needs cmake and the g++ compiler to be compiled.

```bash
    git clone --recursive git@github.com:bonben/galax_eleves.git # clone repository and update git submodules
    cd galax_eleves
    mkdir build # create a build directory
    cd build
    cmake  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" .. # configure project with cmake, using Release build and native arch
    make -j # compile the project
```

You can then launch the compiled binary :
```bash
    ./bin/galax
```

You can access the help with the following command :
```bash
    ./bin/galax -h
```

For now, no graphical display has been set up and therefore only the number of frame per second (which correspond to the number of simulation time steps executed per second) will be displayed in the terminal.

### Graphical display

Galax relies on SDL2 and OpenGL to propose a graphical display.

Now the graphical display should be activated through cmake configuration, and the project be rebuilt.
```bash
    cmake -DGALAX_LINK_SDL2=ON ..
    make -j
    ./bin/galax --display SDL2
```
Launching galax should now open a window with the graphical display.

If dependencies are missing:
```bash
    sudo apt install freeglut3-dev libglew-dev libsdl2-dev
```
### Activating OMP

For now, the performance (FPS: Frame Per Second) is low.
It is possible to accelerate the processing by using OpenMP for multithreading and the xsimd project (https://github.com/xtensor-stack/xsimd) for vectorization.
```bash
    cmake -DGALAX_LINK_OMP=ON ..
    make -j
    ./bin/galax -c CPU_FAST # use CPU_FAST version with multithreading & vectorization
```

### Activating CUDA

Provided a software platform with a working version of the CUDA toolkit, it is also possible to use the GPU to perform the simulation. Please refer to  online documentation to correctly install CUDA on your system (https://docs.nvidia.com/cuda/).

```bash
    cmake -DGALAX_LINK_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="/usr" ..
    make -j
    ./bin/galax -c GPU # use GPU version
```

### Validating your results

To check that the particles positions computed from your code are in agreement with the reference (slow) model, use the `--validate` option. For instance
```bash
    ./bin/galax -c CPU_FAST --validate
```
will compare the positions of the `CPU` and `CPU_FAST` models at each step.

