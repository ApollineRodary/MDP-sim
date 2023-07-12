#include <iostream>
#include <iomanip>
#include <random>
#include "src/algorithms.hpp"
#include "src/io.hpp"
#include "src/mdp/riverswim.cpp"
#include "include/matplotlib-cpp/matplotlibcpp.h"

#define N 8
#define SIM_STEPS 1e7
#define SIM_STEPS_UCRL 5e6

using namespace std;
namespace plt = matplotlibcpp;

int main() {
    auto mdp_info = Riverswim(N, 0.35, 0.05, 0.1, 0.9);
    auto actions = get<0>(mdp_info);
    auto transitions = get<1>(mdp_info);
    auto rewards = get<2>(mdp_info);

    OfflineMDP mdp(actions, transitions, rewards);

    // Find optimal policy and its gain with value iteration algorithm
    cout << "--- Value iteration" << endl;
    auto vi_output = value_iteration(mdp, 1e5, 1e-5);
    Policy policy = get<0>(vi_output);
    double opt_rewards = get<1>(vi_output);
    show_policy(policy);
    cout << "Gain is " << opt_rewards << endl << endl;

    // Apply policy and estimate invariant measure
    cout << "--- Invariant measure" << endl;
    Agent agent(mdp, policy);
    vector<float> d = invariant_measure_estimate(agent, 1e6);
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
    
    // Run UCRL2
    cout << "--- UCRL2" << endl;
    MDP rl_mdp = (MDP) mdp;
    int duration = SIM_STEPS_UCRL;
    auto ucrl_output = ucrl2(rl_mdp, 1e-5, duration);
    History history = ucrl_output.first;
    EpisodeHistory episode_history = ucrl_output.second;

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
    for (Event event: history) {
        i++;
        show_loading_bar("Plotting regret... ", i, history.size());
        
        double reward = get<3>(event);
        total_rl_rewards += reward;
        double regret = i*opt_rewards - total_rl_rewards;
        regrets.push_back(regret);

        int x=get<0>(event), a=get<1>(event);
        total_gap_regret += gap_regret_matrix[x][a];
        gap_regrets.push_back(total_gap_regret);
    }

    cout << "Waiting for matplotlib..." << endl;
    plt::figure();
    plt::plot(regrets);
    plt::plot(gap_regrets);
    plt::save("ucrl2_regret.pdf");

    cout << endl;

    // Look for a bad episode
    cout << "--- Observing gain of an episode with suboptimal history" << endl;
    int k = find_bad_episode(history, episode_history, policy, 1e5);
    int bad_episode_start = episode_history[k].first;
    int bad_episode_duration = episode_history[k+1].first - bad_episode_start;
    cout << "Episode starts at step " << bad_episode_start << " and lasted " << bad_episode_duration << " steps" << endl;
    Policy bad_policy = episode_history[k].second;
    
    History past(history.begin(), history.begin() + bad_episode_start);
    vector<pair<vector<double>, vector<double>>> performance_test_outputs;
    for (int i=0; i<25; i++) {
        show_loading_bar("Performance test... ", i+1, 25);
        auto bad_episode_playback = ucrl2(mdp, 1e-5, 0, 1, past);
        auto performance_test_output = performance_test(mdp, bad_policy, past, get<0>(bad_episode_playback), bad_episode_start, 1000, 1e-5);
        performance_test_outputs.push_back(performance_test_output);
    }

    vector<double> g;
    vector<double> g_opt;
    for (int i=0; i<1000; i++) {
        double gi = 0.0;
        double gi_opt = 0.0;
        for (auto a: performance_test_outputs) {
            gi += a.first[i];
            gi_opt += a.second[i];
        }
        g.push_back(gi/25);
        g_opt.push_back(gi_opt/25);
    }

    plt::figure();
    plt::plot(g_opt);
    plt::plot(g);
    plt::save("performance_test.pdf");

    return 0;
}