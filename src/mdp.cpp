#include <iostream>
#include <iomanip>
#include "mdp.hpp"

MDP::MDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount) {
    this->states = states;
    this->actions = actions;
    this->transitions = transitions;
    this->rewards = rewards;
    this->discount = discount;

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

float MDP::makeAction(int a) {
    // Check if action is available from the current state
    bool valid = false;
    for (int a_: actions[state]) {
        if (a_ == a) {
            valid = true;
            break;
        }
    }
    if (!valid) throw invalid_argument("Illegal action");

    t++;

    // Draw next state
    float *chances = transitions[state][a];
    discrete_distribution<> d{chances, chances+states};
    int next_state = d(gen);

    // Draw rewards (Bernoulli)
    float chance = rewards[state][a];
    float reward = (uniform(gen)<=chance) ? max_reward : 0.0f;

    total_rewards += reward;
    max_reward *= discount;
    state = next_state;
    return reward;
}

int MDP::getStates() {
    return states;
}

int MDP::getState() {
    return state;
}

int MDP::getTime() {
    return t;
}

vector<int> MDP::getAvailableActions() {
    return actions[getState()];
}

float MDP::getDiscount() {
    return discount;
}

OfflineMDP::OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount) : MDP(states, actions, transitions, rewards, discount) {
    // Save separate copy of all hidden pieces of information
    this->actions = actions;
    this->transitions = transitions;
    this->rewards = rewards;
}

vector<int> *OfflineMDP::getActions() {
    /* Get available actions for all states in the MDP */
    return actions;
}

float OfflineMDP::getRewards(int x, int a) {
    /* Get chance of rewards for a given state-action pair */
    return rewards[x][a];
}

float OfflineMDP::getTransitionChance(int x, int a, int y) {
    /* Get chance of transition from state x to state y with action a (i.e. p(y|x,a)) */
    return transitions[x][a][y];
}

float ***OfflineMDP::getTransitionKernel() {
    /* Get transition kernel, i.e. p(y|x,a) for all x, a, y */
    return transitions;
}

vector<int> OfflineMDP::getAvailableActions(int x) {
    /* Get available actions from a given state */
    return actions[x];
}
        
void OfflineMDP::show() {
    /* Display all MDP information */

    // Number of states
    int n = getStates();
    cout << "Showing MDP with " << n << " states" << endl << endl;
    
    // Available actions from every state
    cout << "Actions:" << endl;
    int max_action = 0;
    for (int x=0; x<n; x++) {
        cout << "- " << x << ": ";
        for (int a: getAvailableActions(x)) {
            cout << a << " ";
            if (a > max_action)
                max_action = a;
        }
        cout << endl;
    }
    cout << endl;
    
    // Transition kernel
    cout << "Transitions:" << endl;
    for (int a=0; a<=max_action; a++) {
        cout << "   [Showing transition matrix for action " << a << "]" << endl;
        for(int i=0; i<n; i++) {
            for (int j=0; j<n; j++)
                cout << setw(8) << transitions[i][a][j] << " ";
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    // Chances of rewards for every state-action pair
    cout << "Rewards:" << endl;
    for (int x=0; x<n; x++) {
        cout << "  For state " << x << ": ";
        for (int a=0; a<=max_action; a++)
            cout << setw(8) << rewards[x][a] << " ";
        cout << endl;
    }
    cout << endl;
}

int Policy::operator()(int state, int t) {
    t %= v.size();
    return v[t][state];
}

MDP *Agent::getMDP() {
    return mdp;
}

int Agent::makeRandomAction(float *f) {
    /**
     * Chooses and makes a random action among those available from the current state
     * Saves rewards gotten to f
     * Returns ID of action chosen
     */
    vector<int> actions = mdp->getAvailableActions();
    int action = actions[rand() % actions.size()];
    *f = mdp->makeAction(action);
    return action;
}

int Agent::makeRandomAction() {
    /**
     * Chooses and makes a random action
     * Returns ID of action chosen
     */
    float f;
    return makeRandomAction(&f);
}

int Agent::usePolicy() {
    /**
     * Plays one step of the agent's policy
     * Returns action chosen, whether valid or not
     */
    int state = mdp->getState();
    int t = mdp->getTime();
    int action = (*policy)(state, t);
    mdp->makeAction(action);
    return action;
}

void show_policy(Policy *policy) {
    int steps = size(policy->v);
    if (steps>1)
        cout << "Showing policy with " << size(policy->v) << " steps:" << endl;
    else if (steps==1)
        cout << "Showing stationary policy:";
    else {
        cout << "Asking to show empty policy, discarding";
        return;
    }
    
    int t=0;
    for (auto pol: policy->v) {
        t++;
        cout << "(" << t << "/" << steps << ") ";
        for (int a: pol)
            cout << " " << a;
        cout << endl;
    }
}