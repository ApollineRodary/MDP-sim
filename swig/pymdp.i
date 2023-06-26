%module pymdp

%{
#define SWIG_FILE_WITH_INIT
#include "../src/mdp.hpp"
#include<vector>
#include<random>
%}

using namespace std;

template<typename T>
using Matrix = vector<vector<T>>;

template<typename T>
using Matrix3D = vector<vector<vector<T>>>;

class MDP {
    /**
     *  Markov decision process with hidden information on transitions, actions and rewards, for use in RL
     *  Rewards are Bernoulli
     */

    private:
    Matrix<int> &actions;           // Available actions: actions[x] := vector of actions available from state x
    Matrix3D<float> &transitions;   // Transition kernel: transitions[x][a][y] := p(y | x, a)
    Matrix<float> &rewards;         // Chance for reward: R(x, a) ~ B(rewards[x][a])
    float discount;
    int state;
    int t;
    float max_reward;
    float total_rewards;
    mt19937 gen;
    uniform_real_distribution<> uniform;

    public:
    MDP(Matrix<int> &actions, Matrix3D<float> &transitions, Matrix<float> &rewards, float discount);
    MDP(Matrix<int> &actions, Matrix3D<float> &transitions, Matrix<float> &rewards) : MDP(actions, transitions, rewards, 1.0f) {}
    float makeAction(int action);
    int getState();
    int getStates();
    int getMaxAction();
    int getTime();
    vector<int> &getAvailableActions();
    float getDiscount();
};

class OfflineMDP: public MDP {
    /**
     *  Markov decision process with public information on transitions, actions and rewards
     */

    public:
    Matrix<int> &actions;
    Matrix3D<float> &transitions;
    Matrix<float> &rewards;
    
    OfflineMDP(Matrix<int> &actions, Matrix3D<float> &transitions, Matrix<float> &rewards, float discount) : MDP(actions, transitions, rewards, discount), actions(actions), transitions(transitions), rewards(rewards) {}
    OfflineMDP(Matrix<int> &actions, Matrix3D<float> &transitions, Matrix<float> &rewards) : OfflineMDP(actions, transitions, rewards, 1.0f) {}
    Matrix<int> &getActions();
    float getRewards(int x, int action);
    float getTransitionChance(int x, int action, int y);
    Matrix3D<float> &getTransitionKernel();
    vector<int> &getAvailableActions(int x);
    void show();
};

struct Policy {
    const Matrix<int> v;
    int operator()(int state, int t);
};

class Agent {
    private:
    MDP &mdp;
    Policy &policy;
    
    public:
    Agent(MDP &mdp, Policy &policy) : mdp(mdp), policy(policy) {}
    MDP &getMDP();
    int makeRandomAction(float &f);
    int makeRandomAction();
    int usePolicy();
};

void show_policy(Policy &policy);