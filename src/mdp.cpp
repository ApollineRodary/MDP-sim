#include <iostream>
#include <vector>
#include <random>

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
        float discount = 1;

        int state = 0;
        int t=0;
        float max_reward = 1.0f;
        float total_rewards = 0;

        mt19937 gen;
        uniform_real_distribution<> uniform;

    public:
        MDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount) {
            this->states = states;
            this->actions = actions;
            this->transitions = transitions;
            this->rewards = rewards;
            this->discount = discount;

            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> uniform(0, 1);
            this->gen = gen;
            this->uniform = uniform;
        }

        MDP(int states, vector<int> *actions, float ***transitions, float **rewards) : MDP(states, actions, transitions, rewards, 1.0f) {}

        float makeAction(int a) {
            /* Sends an action to the MDP, provided that the action is legal, and returns instant rewards */

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

        int getStates() {
            // Get number of states of the MDP
            return states;
        }

        int getState() {
            // Get current state
            return state;
        }

        int getTime() {
            // Get number of steps
            return t;
        }

        vector<int> getAvailableActions() {
            // Get available actions from the current state
            return actions[state];
        }

        float getDiscount() {
            return discount;
        }
};

class OfflineMDP: public MDP {
    /**
     *  Markov decision process with public information on transitions, actions and rewards
     */

    public:
        vector<int> *actions;
        float ***transitions;
        float **rewards;

        OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards, float discount) : MDP(states, actions, transitions, rewards, discount) {
            this->actions = actions;
            this->transitions = transitions;
            this->rewards = rewards;
        }

        OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards) : MDP(states, actions, transitions, rewards) {
            this->actions = actions;
            this->transitions = transitions;
            this->rewards = rewards;
        }

        vector<int> *getActions() {
            return actions;
        }

        float getRewards(int x, int a) {
            return rewards[x][a];
        }

        float getTransitionChance(int x, int a, int y) {
            return transitions[x][a][y];
        }

        float ***getTransitionKernel() {
            return transitions;
        }

        vector<int> getAvailableActions() {
            // Get available actions from the current state
            return actions[getState()];
        }

        vector<int> getAvailableActions(int x) {
            // Get available actions from a given state
            return actions[x];
        }
};

struct Policy {
    const vector<vector<int>> v;

    int operator()(int state, int t) {
        t %= v.size();
        return v[t][state];
    }
};

class Agent {
    private:
        MDP *mdp;
        Policy *policy;
    
    public:
        Agent(MDP *mdp) : mdp(mdp) {}
        Agent(MDP *mdp, Policy *policy) : mdp(mdp), policy(policy) {}

        MDP *getMDP() {
            return mdp;
        }

        int makeRandomAction(float *f) {
            vector<int> actions = mdp->getAvailableActions();
            int action = actions[rand() % actions.size()];
            *f = mdp->makeAction(action);
            return action;
        }

        int makeRandomAction() {
            float f;
            return makeRandomAction(&f);
        }

        int usePolicy() {
            int state = mdp->getState();
            int t = mdp->getTime();
            int action = (*policy)(state, t);
            mdp->makeAction(action);
            return action;
        }
};

void print_policy(Policy *policy) {
    cout << "Policy has " << size(policy->v) << " steps:" << endl;
    int t=0;
    for (auto pol: policy->v) {
        t++;
        cout << "(" << t << "/" << size(policy->v) << ") ";
        for (int a: pol) cout << " " << a;
        cout << endl;
    }
}

vector<float> stationary_distribution(Agent *agent, int steps) {
    /* Empirically finds stationary distribution */
    
    vector<float> frequency;
    for (int x=0; x<agent->getMDP()->getStates(); x++) frequency.push_back(0);

    for (int i=0; i<steps; i++) {
        agent->usePolicy();
        frequency[agent->getMDP()->getState()]++;
    }
    
    vector<float> d;
    for (int f: frequency) d.push_back(((float) f)/steps);   
    return d;
}