#include <iomanip>
#include "../src/mdp.cpp"
#define LEFT 0
#define RIGHT 1
#define N 4

int main() {
    /*
        Expected: always go to the right in the default version, even if it isn't immediately profitable (works)
        Variant: boost reward of going left from state 0 way up - we'd expect that going left from anywhere is now the optimal policy (doesn't work: on state N-1, we prefer going to the right still)
    */
    vector<int> actions[N];
    for (int i=0; i<N; i++) actions[i] = {LEFT, RIGHT};

    float ***transitions = new float**[N];
    for (int x=0; x<N; x++) {
        transitions[x] = new float*[2];
        transitions[x][LEFT] = new float[N];
        transitions[x][RIGHT] = new float[N];
        for (int y=0; y<N; y++) {
            transitions[x][LEFT][y] = 0.0;
            transitions[x][RIGHT][y] = 0.0;
        }
        if (x>0 && x<N-1) {
            transitions[x][RIGHT][x+1] = 0.35;
            transitions[x][RIGHT][x] = 0.6;
            transitions[x][RIGHT][x-1] = 0.05;
            transitions[x][LEFT][x-1] = 1.0;
        }        
    }
    transitions[0][RIGHT][0] = 0.6;
    transitions[0][RIGHT][1] = 0.4;
    transitions[0][LEFT][0] = 1.0;
    transitions[N-1][RIGHT][N-1] = 0.95;
    transitions[N-1][RIGHT][N-2] = 0.05;
    transitions[N-1][LEFT][N-2] = 1.0;

    float **rewards = new float*[N];
    for (int x=0; x<N; x++) {
        rewards[x] = new float[2];
        rewards[x][LEFT] = 0.0;
        rewards[x][RIGHT] = 0.0;
    }
    rewards[0][LEFT] = 0.1;
    rewards[N-1][RIGHT] = 0.9;

    // Display transition matrices

    cout << "Left: " << endl;
    for(int i=0; i<N; i++) {
        for (int j=0; j<N; j++) cout << setw(10) << transitions[i][LEFT][j] << " ";
        cout << endl;
    }
    cout << endl;

    cout << "Right: " << endl;
    for(int i=0; i<N; i++) {
        for (int j=0; j<N; j++) cout << setw(10) << transitions[i][RIGHT][j] << " ";
        cout << endl;
    }
    cout << endl;

    // Find and display optimal policy with value iteration algorithm

    MDP mdp(N, actions, transitions, rewards);
    Policy policy = value_iteration(&mdp, N, 1e9, 1e-6);
    print_policy(&policy);

    // Apply policy and estimate stationary distribution

    Agent agent(&mdp, &policy);
    vector<float> d = stationary_distribution(&agent, 1e6);
    for (float f: d) {
        cout << f << " ";
    }

    return 0;
}