#include <iostream>
#include <vector>
#include <random>

using namespace std;

class MDP {
    public:
    int states;
    vector<int> *actions;
    float ***transitions;
    float **rewards;
    float discount = 1;

    int state = 0;
    int t=0;
    float total_rewards = 0;
    mt19937 gen;
    uniform_real_distribution<> uniform;

    MDP(int states, vector<int> *actions, float ***transitions) {}

    MDP(int states, vector<int> *actions, float ***transitions, float **rewards) {
        this->states = states;
        this->actions = actions;
        this->transitions = transitions;
        this->rewards = rewards;

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> uniform(0, 1);
        this->gen = gen;
        this->uniform = uniform;
    }

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

        // Draw rewards (Bernoulli)
        float chance = rewards[state][a];
        if (uniform(gen) <= chance)
            total_rewards += 1.0f;      // Keeping this as a float for when I implement discounts

        // Draw next state
        float *chances = transitions[state][a];
        discrete_distribution<> d{chances, chances+states};
        state = d(gen);
        return rewards[state][a];
    }

    int getState() {
        return state;
    }

    int getTime() {
        return t;
    }

    vector<int> getAvailableActions() {
        return actions[state];
    }

    vector<int> getAvailableActions(int x) {
        return actions[x];
    }

    float getTotalRewards() {
        return total_rewards;
    }
    
    void setState(int state) {
        if (state<0 || state>=states) throw invalid_argument("State must be between 0 and n");
        this->state = state;
    }

    void setDiscount(float discount) {
        if (discount<=0 || discount>1) throw invalid_argument("Discount must be within (0,1]");
        this->discount = discount;
    }

    void reset() {
        setState(0);
        total_rewards = 0;
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

        int makeRandomAction() {
            vector<int> actions = mdp->getAvailableActions();
            int action = actions[rand() % actions.size()];
            mdp->makeAction(action);
            return action;
        }

        int usePolicy() {
            int state = mdp->getState();
            int t = mdp->getTime();
            int action = (*policy)(state, t);
            mdp->makeAction(action);
            return action;
        }
};

Policy value_iteration(MDP *mdp, int n, int max_steps, float eps, vector<float> *d) {
    /* 
        Runs value iteration on an MDP with n states until the span of the difference gets lower than eps
        Returns corresponding policy and sends corresponding stationary distribution to d
    */

    if (eps<=0) throw invalid_argument("eps must be a positive value");

    float v[n];
    float w[n];
    int best_action[n];
    
    for (int i=0; i<n; i++) v[i] = 0.0;

    for (int t=0; t<max_steps; t++) {
        // Compute w out of v (Bellman equation)
        for (int x=0; x<n; x++) {
            vector<int> actions = mdp->getAvailableActions(x);
            float max_q = 0;
            for (int a: actions) {
                // q = Q_{t+1}*(x, a)
                float q = mdp->rewards[x][a];
                for (int y=0; y<n; y++) q += mdp->transitions[x][a][y] * v[y];
                if (q>max_q) {
                    // max_q =          max_{a \in A(x)} Q_{t+1}*(x, a)
                    // best_action[x] = argmax of above
                    max_q = q;
                    best_action[x] = a;
                };
            }
            w[x] = max_q;
        }

        float max_dv = -INFINITY;
        float min_dv = INFINITY;
        for (int x=0; x<n; x++) {
            float dv = w[x]-v[x];
            if (dv > max_dv) max_dv = dv;
            if (dv < min_dv) min_dv = dv;
            v[x] = w[x];
        }

        if (max_dv - min_dv < eps) {
            vector<int> pol;
            for (int x=0; x<n; x++) {
                pol.push_back(best_action[x]);
                d->push_back(v[x]);
            }
            Policy policy = {{pol}};
            return policy;
        }
    }
    Policy policy = {{}};
    return policy;
}


Policy value_iteration(MDP *mdp, int n, int max_steps, float eps) {
    vector<float> d;
    return (value_iteration(mdp, n, max_steps, eps, &d));
}

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
    for (int x=0; x<agent->getMDP()->states; x++) frequency.push_back(0);

    for (int i=0; i<steps; i++) {
        agent->usePolicy();
        frequency[agent->getMDP()->getState()]++;
    }
    
    vector<float> d;
    for (int f: frequency) d.push_back(((float) f)/steps);   
    return d;
}