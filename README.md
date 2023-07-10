# MDP-Sim

Libraries and simulations about [Markov decision processes](https://en.wikipedia.org/wiki/Markov_decision_process) (MDPs) for an internship at [Laboratoire d'Informatique de Grenoble](https://www.liglab.fr).

## Building

C++ executables for tests can be built using CMake with the following commands (Linux):

```sh
mkdir build
cd build
cmake ..
make
```

Executables are built in the `build` directory.
It is assumed that matplotlib-cpp, numpy and Python are installed, with matplotlib-cpp placed in the `include` directory.

The `pymdp` Python library can be built using SWIG. The `build_pylibs.sh` script automates this process.
The library is built in the `swig` directory.
It is assumed that Python development headers are installed.

## Contents

The project contains C++ headers to represent and simulate Markov decision processes, offline, or for reinforcement learning, with functions for:

- Getting a near-optimal policy on average with value iteration
- Estimating a policy's invariant measure
- Getting a policy's invariant measure with value iteration
- Running UCRL2 on an MDP and getting the resulting history
- Getting regret and gap-regret from a history of plays on an MDP
