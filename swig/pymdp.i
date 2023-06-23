%module pymdp

%{
#define SWIG_FILE_WITH_INIT
#include "../src/mdp.hpp"
#include<vector>
#include<random>
%}

using namespace std;

class MDP {
    /**
     *  Markov decision process with hidden information on transitions, actions and rewards, for use in RL
     *  Rewards are Bernoulli
     */

    private:
    int states;
    vector<int> *actions;       // Available actions: actions[x] := vector of actions available from state x
    float ***transitions;       // Transition kernel: transitions[x][a][y] := p(y | x, a)
    float **rewards;            // Chance for reward: R(x, a) ~ B(rewards[x][a])
    float discount;
    int state;
    int t;
    float max_reward;
    float total_rewards;
    mt19937 gen;
    uniform_real_distribution<> uniform;

    public:
    MDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount);
    MDP(int states, vector<int> *actions, float ***transitions, float **rewards) : MDP(states, actions, transitions, rewards, 1.0f) {}
    float makeAction(int a);
    int getStates();
    int getState();
    int getTime();
    vector<int> getAvailableActions();
    float getDiscount();
};

class OfflineMDP: public MDP {
    /**
     *  Markov decision process with public information on transitions, actions and rewards
     */

    public:
    vector<int> *actions;
    float ***transitions;
    float **rewards;
    OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount);
    OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards) : OfflineMDP(states, actions, transitions, rewards, 1.0f) {}
    vector<int> *getActions();
    float getRewards(int x, int a);
    float getTransitionChance(int x, int a, int y);
    float ***getTransitionKernel();
    vector<int> getAvailableActions(int x);
    void show();
};

struct Policy {
    const vector<vector<int>> v;
    int operator()(int state, int t);
};

class Agent {
    private:
    MDP *mdp;
    Policy *policy;
    
    public:
    Agent(MDP *mdp) : mdp(mdp) {}
    Agent(MDP *mdp, Policy *policy) : mdp(mdp), policy(policy) {}
    MDP *getMDP();
    int makeRandomAction(float *f);
    int makeRandomAction();
    int usePolicy();
};

void show_policy(Policy *policy);