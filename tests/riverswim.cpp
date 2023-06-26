#include <iostream>
#include <iomanip>
#include "src/algorithms.hpp"

#define LEFT 0
#define RIGHT 1
#define N 10
#define SIM_STEPS 1e6

using namespace std;

int main() {
    Matrix<int> actions(N, {LEFT, RIGHT});
    Matrix3D<float> transitions(N, Matrix<float>(2, vector<float>(N, 0.0)));

    for (int x=1; x<N-1; x++) {
        transitions[x][RIGHT][x+1] = 0.35;
        transitions[x][RIGHT][x] = 0.6;
        transitions[x][RIGHT][x-1] = 0.05;
        transitions[x][LEFT][x-1] = 1.0;
    }
    transitions[0][RIGHT][0] = 0.6;
    transitions[0][RIGHT][1] = 0.4;
    transitions[0][LEFT][0] = 1.0;
    transitions[N-1][RIGHT][N-1] = 0.95;
    transitions[N-1][RIGHT][N-2] = 0.05;
    transitions[N-1][LEFT][N-2] = 1.0;

    Matrix<float> rewards(N, {0.0, 0.0});
    rewards[0][LEFT] = 0.1;
    rewards[N-1][RIGHT] = 0.9;

    // Display transition matrices
    OfflineMDP mdp(actions, transitions, rewards, 1.0f);

    cout << "Left: " << endl;
    for(int i=0; i<N; i++) {
        for (int j=0; j<N; j++) cout << setw(10) << mdp.getTransitionChance(i, LEFT, j) << " ";
        cout << endl;
    }
    cout << endl;

    cout << "Right: " << endl;
    for(int i=0; i<N; i++) {
        for (int j=0; j<N; j++) cout << setw(10) << mdp.getTransitionChance(i, RIGHT, j) << " ";
        cout << endl;
    }
    cout << endl;

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

    return 0;
}