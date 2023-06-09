#include <vector>
#include "../src/mdp.cpp"

using namespace std;

int main() {
    // A weird example: 10 states, actions are numbers coprime with the current state, and picking an action is adding the action number to that state with probability 0.91.
    int states = 10;
    vector<int> actions[10];
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
            rewards[x][a] = (float)((x+a)%10) / 10;
        }
    }

    MDP mdp(states, actions, transitions, rewards);
    Agent agent(&mdp);

    int steps = 10;
    for (int i=1; i<=steps; i++) {
        cout << "Step (" << i << "/" << steps << ")" << endl;
        int stateBefore = mdp.getState();
        cout << "  Current state: " << stateBefore + 1 << endl;
        cout << "  Available actions:";
        for (int a: mdp.getAvailableActions()) cout << " " << a+1;
        float rewardBefore = mdp.getTotalRewards();
        int action = agent.makeRandomAction();
        cout << endl << "  Choosing action " << action + 1 << endl;
        float rewardAfter = mdp.getTotalRewards();
        cout << "  Action was rewarded with " << rewardAfter - rewardBefore << endl;
        cout << "=============================================" << endl;
    }
}