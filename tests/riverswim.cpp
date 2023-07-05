#include <iostream>
#include <iomanip>
#include <random>
#include "src/algorithms.hpp"
#include "src/mdp/riverswim.cpp"

#define N 3
#define SIM_STEPS 1e6

using namespace std;

int main() {
    auto mdp_info = Riverswim(N, 0.35, 0.05, 0.1, 0.9);
    auto actions = get<0>(mdp_info);
    auto transitions = get<1>(mdp_info);
    auto rewards = get<2>(mdp_info);

    OfflineMDP mdp(actions, transitions, rewards);

    // Find and display optimal policy with value iteration algorithm
    Policy policy = value_iteration(mdp, 1e6, 1e-6);
    show_policy(policy);
    cout << endl;

    // Apply policy and estimate invariant measure
    Agent agent(mdp, policy);
    vector<float> d = invariant_measure_estimate(agent, SIM_STEPS);
    cout << "Invariant measure after " << SIM_STEPS << " steps is estimated to be:" << endl;
    for (float f: d)
        cout << setw(12) << f << " ";
    cout << endl << endl;

    // Get invariant measure from value iteration
    vector<float> im = invariant_measure(mdp, policy);
    cout << "Invariant measure with value iteration is supposed to be:" << endl;
    for (float f: im)
        cout << setw(12) << f << " ";
    cout << endl << endl;

    // Get gain from invariant measure
    double opt_rewards = 0.0;
    for (int i=0; i<N; i++)
        opt_rewards += im[i]*mdp.getRewards(i, policy(i, 0));
    
    // Run UCRL2
    MDP rl_mdp(actions, transitions, rewards);
    int duration = 100000;
    vector<double> rl_rewards = ucrl2(rl_mdp, 0.01, duration);

    int i=0;
    for (double r: rl_rewards) {
        i++;
        cout << setw(10) << opt_rewards*i*1000 - r;
    }
    cout << endl;

    return 0;
}