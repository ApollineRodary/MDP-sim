#include <vector>
#include "src/mdp.hpp"

using namespace std;

int main() {
    // A weird example: 10 states, actions are numbers coprime with the current state, and picking an action is adding the action number to that state with probability 0.91.
    int states = 10;
    vector<int> *actions = new vector<int>[10];
    for (int i=0; i<10; i++) {
        actions[i] = {};
        for (int j=0; j<10; j++) {
            if (gcd(i+1, j+1) == 1) actions[i].push_back(j);
        }
    }
    float ***transitions = new float**[10];
    for (int x=0; x<10; x++) {
        transitions[x] = new float*[10];
        for (int a=0; a<10; a++) {
            transitions[x][a] = new float[10];
            for (int y=0; y<10; y++) transitions[x][a][y] = 0.01f;
            transitions[x][a][(x+a+1)%10] = 0.91f;
        }
    }
    float **rewards = new float*[10];
    for (int x=0; x<10; x++) {
        rewards[x] = new float[10];
        for (int a=0; a<10; a++) {
            rewards[x][a] = (float)((x+a+2)%10) / 10;
        }
    }

    MDP mdp(states, actions, transitions, rewards);
    Agent agent(&mdp);
    vector<int> available = mdp.getAvailableActions();
    for (int a: available) cout << a << " ";
    cout << endl;

    int steps = 30;
    for (int i=1; i<=steps; i++) {
        // Show current step
        cout << "Step (" << i << "/" << steps << ")" << endl;
        
        // Show current state and available actions
        int stateBefore = mdp.getState();
        cout << "  Current state: " << stateBefore + 1 << endl;
        cout << "  Available actions:";
        for (int a: mdp.getAvailableActions()) cout << " " << a+1;

        // Run random action and save reward
        float reward;
        int action = agent.makeRandomAction(&reward);
        
        // Show the action that was chosen and the reward we got
        cout << endl << "  Choosing action " << action + 1 << endl;
        cout << "  Action was rewarded with " << reward << endl;

        cout << "=============================================" << endl;
    }
}