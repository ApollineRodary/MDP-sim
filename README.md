# MDP-Sim

Code for dealing with Markov decision processes (MDPs) for an internship at Laboratoire d'Informatique de Grenoble

## Contents

The project contains C++ headers to represent and simulate Markov decision processes, offline, or for reinforcement learning, with functions for:
- Getting a near-optimal policy on average with value iteration
- Estimating a policy's invariant measure
- Getting a policy's invariant measure with value iteration

C++ DLLs for use in Python with Boost.Python are also available, and used in Python simulations with Tkinter.

## Documentation

An MDP is represented as a tuple `(S, A, P, R, d)` where:
- `S` is a set of states
- `A` is a set of actions; in particular, for `x` in `S`, `A(x)` is the set of actions available from state `x`
- `P` is the transition kernel: action `a` in `A(x)` moves from state `x` to state `y` with probability `p(y|x,a) := P[x][a][y]`
- `R` is the reward matrix: choosing action `a` in `A(x)` from state `x` draws rewards according to Bernoulli distribution of parameter `R[x][a]`
- `d` is the discount: rewards on the `n`th simulation step are multiplied by `d^n`

The `MDP` class defined in `src/mdp.hpp` is defined with a number of states, an array of `vector<int>` s representing allowed states from every state, and a transition kernel, a reward matrix, and a discount as defined above.