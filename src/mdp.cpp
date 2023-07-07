#include <iostream>
#include <iomanip>
#include "mdp.hpp"

MDP::MDP(Matrix<int> &actions, Matrix3D<float> &transitions, Matrix<float> &rewards, float discount) : actions(actions), transitions(transitions), rewards(rewards), discount(discount) {
    max_reward = 1.0f;
    state = 0;
    t = 0;
    total_rewards = 0;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> uniform(0, 1);
    this->gen = gen;
    this->uniform = uniform;
}

float MDP::makeAction(int action) {
    // Check if action is available from the current state
    bool valid = false;
    for (int action_: actions[state]) {
        if (action_ == action) {
            valid = true;
            break;
        }
    }
    if (!valid) throw invalid_argument("Illegal action");

    t++;

    // Draw next state
    vector<float> chances = transitions[state][action];
    discrete_distribution<> d{chances.begin(), chances.end()};
    int next_state = d(gen);

    // Draw rewards (Bernoulli)
    float chance = rewards[state][action];
    float reward = (uniform(gen)<=chance) ? max_reward : 0.0f;

    total_rewards += reward;
    max_reward *= discount;
    state = next_state;
    return reward;
}

int MDP::getState() {
    return state;
}

int MDP::getStates() {
    return transitions.size();
}

int MDP::getMaxAction() {
    return transitions[0].size();
}

int MDP::getTime() {
    return t;
}

vector<int> &MDP::getAvailableActions() {
    return actions[getState()];
}

vector<int> &MDP::getAvailableActions(int x) {
    return actions[x];
}

Matrix<int> &MDP::getActions() {
    return actions;
}


float MDP::getDiscount() {
    return discount;
}

float OfflineMDP::getRewards(int x, int action) {
    /* Get chance of rewards for a given state-action pair */
    return rewards[x][action];
}

float OfflineMDP::getTransitionChance(int x, int action, int y) {
    /* Get chance of transition from state x to state y with action a (i.e. p(y|x,a)) */
    int n = getStates();
    int a = getMaxAction();
    if (x<0 || x>=n || y<0 || y>=n || action<0 || action>=a)
        throw invalid_argument("bruh");
    return transitions[x][action][y];
}

Matrix3D<float> &OfflineMDP::getTransitionKernel() {
    /* Get transition kernel, i.e. p(y|x,a) for all x, a, y */
    return transitions;
}

void OfflineMDP::show() {
    /* Display all MDP information */

    // Number of states/actions
    int n = getStates();
    int a = getMaxAction();
    cout << "Showing MDP with " << n << " states and " << a << " actions" << endl << endl;
    
    // Available actions from every state
    cout << "Actions:" << endl;
    int max_action = 0;
    for (int x=0; x<n; x++) {
        cout << "- " << x << ": ";
        for (int action: getAvailableActions(x)) {
            cout << action << " ";
            if (action > max_action)
                max_action = action;
        }
        cout << endl;
    }
    cout << endl;
    
    // Transition kernel
    cout << "Transitions:" << endl;
    for (int action=0; action<=max_action; action++) {
        cout << "   [Showing transition matrix for action " << action << "]" << endl;
        for(int i=0; i<n; i++) {
            for (int j=0; j<n; j++)
                cout << setw(8) << getTransitionChance(i, action, j) << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    // Chances of rewards for every state-action pair
    cout << "Rewards:" << endl;
    for (int x=0; x<n; x++) {
        cout << "  For state " << x << ": ";
        for (int action=0; action<=max_action; action++)
            cout << setw(8) << getRewards(x, action) << " ";
        cout << endl;
    }
    cout << endl;
}

int Policy::operator()(int state, int t) {
    t %= v.size();
    return v[t][state];
}

MDP &Agent::getMDP() {
    return mdp;
}

int Agent::makeRandomAction(float &f) {
    /**
     * Chooses and makes a random action among those available from the current state
     * Saves rewards to f
     * Returns ID of action chosen
     */
    vector<int> actions = mdp.getAvailableActions();
    int action = actions[rand() % actions.size()];
    f = mdp.makeAction(action);
    return action;
}

int Agent::makeRandomAction() {
    /**
     * Chooses and makes a random action
     * Returns ID of action chosen
     */
    float f;
    return makeRandomAction(f);
}

int Agent::usePolicy(float &f) {
    /**
     * Plays one step of the agent's policy
     * Saves rewards to f
     * Returns action chosen, whether valid or not
     */
    int state = mdp.getState();
    int t = mdp.getTime();
    int action = policy(state, t);
    f = mdp.makeAction(action);
    return action;
}

int Agent::usePolicy() {
    /**
     * Plays one step of the agent's policy
     * Returns action chosen
     */
    float f;
    return usePolicy(f);
}

void show_policy(Policy &policy) {
    int steps = size(policy.v);
    if (steps>1)
        cout << "Showing policy with " << size(policy.v) << " steps:" << endl;
    else if (steps==1)
        cout << "Showing stationary policy: ";
    else {
        cout << "Asking to show empty policy, discarding";
        return;
    }
    
    int t=0;
    for (auto pol: policy.v) {
        t++;
        cout << "(" << t << "/" << steps << ") ";
        for (int a: pol)
            cout << " " << a;
        cout << endl;
    }
}