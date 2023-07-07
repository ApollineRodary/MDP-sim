#include <iostream>
#include <iomanip>
#include <random>
#include "src/algorithms.hpp"
#include "src/io.hpp"
#include "src/mdp/riverswim.cpp"
#include "include/matplotlib-cpp/matplotlibcpp.h"

#define N 5
#define SIM_STEPS 1e5
#define SIM_STEPS_UCRL 1e7

using namespace std;
namespace plt = matplotlibcpp;

int main() {
    auto mdp_info = Riverswim(N, 0.35, 0.05, 0.1, 0.9);
    auto actions = get<0>(mdp_info);
    auto transitions = get<1>(mdp_info);
    auto rewards = get<2>(mdp_info);

    OfflineMDP mdp(actions, transitions, rewards);

    // Find and display optimal policy with value iteration algorithm
    Policy policy = get<0>(value_iteration(mdp, 1e5, 1e-5));
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
    cout << "That's a gain of " << opt_rewards << endl << endl;
    
    // Run UCRL2
    MDP rl_mdp(actions, transitions, rewards);
    int duration = SIM_STEPS_UCRL;
    History ucrl_history = ucrl2(rl_mdp, 1e-5, duration);

    // Compute gap regrets
    double total_rl_rewards=0, total_gap_regret=0;
    Matrix<double> gap_regret_matrix(N, vector<double>(2));
    for (int x=0; x<N; x++)
        for (int a=0; a<1; a++)
            gap_regret_matrix[x][a] = gap_regret(x, a, mdp);
    
    // Plot empirical regrets and gap regrets
    vector<double> regrets;
    vector<double> gap_regrets;
    
    int i=0;
    for (Event event: ucrl_history) {
        i++;
        show_loading_bar("Plotting regret...", i, ucrl_history.size());
        
        double reward = get<2>(event);
        total_rl_rewards += reward;
        double regret = i*opt_rewards - total_rl_rewards;
        regrets.push_back(regret);

        int x=get<0>(event), a=get<1>(event);
        total_gap_regret += gap_regret_matrix[x][a];
        gap_regrets.push_back(total_gap_regret);
    }

    plt::plot(regrets);
    plt::plot(gap_regrets);
    plt::save("ucrl2_regret.pdf");

    cout << endl;

    return 0;
}