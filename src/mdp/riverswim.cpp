#include "../mdp.hpp"
#include <tuple>

#define LEFT 0
#define RIGHT 1
using namespace std;

tuple<Matrix<int>, Matrix3D<float>, Matrix<float>> Riverswim(int n, float progress_chance, float flow_back_chance, float lazy_reward, float win_reward) {
    float halt_chance = 1.0 - progress_chance - flow_back_chance;
    
    Matrix<int> actions(n, {LEFT, RIGHT});
    Matrix3D<float> transitions(n, Matrix<float>(2, vector<float>(n, 0.0)));
    
    for (int x=1; x<n-1; x++) {
        transitions[x][RIGHT][x+1] = progress_chance;
        transitions[x][RIGHT][x] = halt_chance;
        transitions[x][RIGHT][x-1] = flow_back_chance;
        transitions[x][LEFT][x-1] = 1.0;
    }
    transitions[0][RIGHT][0] = halt_chance;
    transitions[0][RIGHT][1] = progress_chance + flow_back_chance;
    transitions[0][LEFT][0] = 1.0;
    transitions[n-1][RIGHT][n-1] = progress_chance + halt_chance;
    transitions[n-1][RIGHT][n-2] = flow_back_chance;
    transitions[n-1][LEFT][n-2] = 1.0;

    Matrix<float> rewards(n, {0.0, 0.0});
    rewards[0][LEFT] = lazy_reward;
    rewards[n-1][RIGHT] = win_reward;

    return tuple(actions, transitions, rewards);
}