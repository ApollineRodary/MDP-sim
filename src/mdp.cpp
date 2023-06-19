#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

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

            // Init RNG
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> uniform(0, 1);
            this->gen = gen;
            this->uniform = uniform;
        }

        MDP(int states, vector<int> *actions, float ***transitions, float **rewards) : MDP(states, actions, transitions, rewards, 1.0f) {}

        float makeAction(int a) {
            /* Sends an action to the MDP, provided that the action is legal, and returns instant rewards */

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

        int getStates() {
            /* Get number of states of the MDP */
            return states;
        }

        int getState() {
            /* Get current state */
            return state;
        }

        int getTime() {
            /* Get number of steps so far */
            return t;
        }

        vector<int> getAvailableActions() {
            /* Get actions available from the current state */
            return actions[getState()];
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
            // Save separate copy of all hidden pieces of information
            this->actions = actions;
            this->transitions = transitions;
            this->rewards = rewards;
        }

        OfflineMDP(int states, vector<int> *actions, float ***transitions, float **rewards) : OfflineMDP(states, actions, transitions, rewards, 1.0f) {}

        vector<int> *getActions() {
            /* Get available actions for all states in the MDP */
            return actions;
        }

        float getRewards(int x, int a) {
            /* Get chance of rewards for a given state-action pair */
            return rewards[x][a];
        }

        float getTransitionChance(int x, int a, int y) {
            /* Get chance of transition from state x to state y with action a (i.e. p(y|x,a)) */
            return transitions[x][a][y];
        }

        float ***getTransitionKernel() {
            /* Get transition kernel, i.e. p(y|x,a) for all x, a, y */
            return transitions;
        }

        vector<int> getAvailableActions(int x) {
            /* Get available actions from a given state */
            return actions[x];
        }
        
        void show() {
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

        int makeRandomAction() {
            /**
             * Chooses and makes a random action
             * Returns ID of action chosen
             */
            float f;
            return makeRandomAction(&f);
        }

        int usePolicy() {
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
};

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

vector<float> invariant_measure_estimate(Agent *agent, int steps) {
    /**
     * Get empirical estimate of invariant measure
     * Agent uses its policy on its MDP starting from the MDP's state when calling the function
     * Return value is frequency of visit of every state
     */
    
    vector<float> frequency;
    for (int x=0; x<agent->getMDP()->getStates(); x++)
        frequency.push_back(0);

    for (int i=0; i<steps; i++) {
        agent->usePolicy();
        frequency[agent->getMDP()->getState()]++;
    }
    
    vector<float> d;
    for (int f: frequency)
        d.push_back(((float) f)/steps);   
    return d;
}