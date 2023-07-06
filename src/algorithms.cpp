#include <stdexcept>
#include <algorithm>
#include "algorithms.hpp"
#include <iostream>
#include <iomanip>

Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps, float &g) {
    /* 
        Runs value iteration on an MDP with n states until the span of the difference gets lower than eps
        Returns corresponding policy and sends corresponding gain to g
    */

    if (eps<=0)
        throw invalid_argument("eps must be a positive value");

    int n = mdp.getStates();
    int a = mdp.getMaxAction();

    float v[n];
    float w[n];
    int best_action[n];
    
    for (int i=0; i<n; i++) v[i] = 0.0;

    for (int t=0; t<max_steps; t++) {
        // Compute w out of v (Bellman equation)
        for (int x=0; x<n; x++) {
            vector<int> actions = mdp.getAvailableActions(x);
            float max_q = -INFINITY;
            for (int action: actions) {
                // q = Q_{t+1}*(x, a)
                float q = mdp.getRewards(x, action);
                for (int y=0; y<n; y++)
                    q += mdp.getTransitionChance(x, action, y) * v[y];
                if (q>max_q) {
                    // max_q =          max_{a \in A(x)} Q_{t+1}*(x, a)
                    // best_action[x] = argmax of above
                    max_q = q;
                    best_action[x] = action;
                };
            }
            w[x] = max_q;
        }

        float max_dv = -INFINITY;
        float min_dv = INFINITY;
        for (int x=0; x<n; x++) {
            float dv = w[x]-v[x];
            if (dv > max_dv)
                max_dv = dv;
            if (dv < min_dv)
                min_dv = dv;
            v[x] = w[x];
        }
        float v0 = v[0];
        for (int x=0; x<n; x++)
            v[x] -= v0;
        
        if (max_dv - min_dv < eps) {
            vector<int> pol;
            g = (max_dv + min_dv)/2;
            for (int x=0; x<n; x++)
                pol.push_back(best_action[x]);
            Policy policy = {{pol}};
            return policy;
        }
    }
    Policy policy = {{}};
    return policy;
}

Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps) {
    float g;
    return (value_iteration(mdp, max_steps, eps, g));
}

vector<float> invariant_measure(OfflineMDP &mdp, Policy &policy) {
    /* Get invariant measure of a policy with value iteration */
    
    int n = mdp.getStates();
    int a = mdp.getMaxAction();
    vector<float> ans;
    
    for (int x=0; x<n; x++) {
        // Only policy actions are legal
        Matrix<int> actions;
        for (int i=0; i<n; i++) 
            actions.push_back({policy(i, 0)});

        // Every action awards 0 except from state x
        Matrix<float> rewards(n, vector<float>(a, 0.0));
        rewards[x][policy(x, 0)] = 1.0;

        float g;
        OfflineMDP nmdp = OfflineMDP(actions, mdp.getTransitionKernel(), rewards);
        value_iteration(nmdp, 1e6, 1e-6, g);
        ans.push_back(g);
    }
    return ans;
}

vector<float> invariant_measure_estimate(Agent &agent, int steps) {
    /**
     * Get empirical estimate of invariant measure
     * Agent uses its policy on its MDP starting from the MDP's state when calling the function
     * Return value is frequency of visit of every state
     */
    
    vector<float> frequency;
    for (int x=0; x<agent.getMDP().getStates(); x++)
        frequency.push_back(0);

    for (int i=0; i<steps; i++) {
        agent.usePolicy();
        frequency[agent.getMDP().getState()]++;
    }
    
    vector<float> d;
    for (int f: frequency)
        d.push_back(((float) f)/steps);   
    return d;
}

double optimize(vector<double> &p, vector<double> &u, double eps) {
    /**
     * Solves the following optimization problem:
     * Find vector q that maximizes < q | u > under the constraints
     *  . |p-q| < eps, where |.| is 1-norm
     *  . |q| = 1
     *  . 0 <= q(x) <= 1 for all x
     * Returns < q | u >
     */

    int n = p.size();

    // Sort states descendingly according to their u-values
    vector<int> s(n);
    for (int i=0; i<n; i++)
        s[i] = i;
    stable_sort(s.begin(), s.end(), [&](int i, int j) {return u[i] > u[j];});

    // Add as much weight as possible to q_i with i maximizing u_i
    vector<double> q(n);
    for (int i=0; i<n; i++)
        q[s[i]] = p[s[i]];
    
    // Remove as much weight as possible to q_j with j minimizing u_j until it evens out 
    int i=0, j=n-1;
    while (i<j) {
        float m = 0.5*eps;
        if (1.0-q[s[i]] < m) m = 1.0 - q[s[i]];
        if (q[s[j]] < m) m = q[s[j]];

        q[s[i]] += m;
        q[s[j]] -= m;
        eps -= 2*m;
        if (m == eps*0.5)
            break;
        if (m == 1.0-q[s[i]])
            i++;
        else
            j--;
    }

    return inner_product(q.begin(), q.end(), u.begin(), 0.0);
}

Policy extended_value_iteration(MDP &mdp, Matrix3D<double> &estimated_transition_chances, Matrix<double> &estimated_rewards, Matrix<double> &transition_chance_uncertainty, Matrix<double> &reward_uncertainty, int max_steps, float eps) {
    /**
     * Runs extended value iteration until span of u-value is below eps and returns corresponding policy
     * Extended MDP has:
     *  - states as in mdp,
     *  - transitions p within ||p[x][a] - estimated_transition_chances[x][a][.]|| < transition_chance_uncertainty[x][a],
     *  - rewards within estimated_rewards +/- reward_uncertainty
     * Computation of inner maximum according to NEAR-OPTIMAL REGRET BOUNDS FOR REINFORCEMENT LEARNING, Jaksch & al
     */

    int n = mdp.getStates();
    int a = mdp.getMaxAction();

    vector<double> v(n, 0.0);
    vector<double> w(n);
    vector<int> best_action(n);
    
    for (int t=0; t<max_steps; t++) {
        for (int x=0; x<n; x++) {
            float max_q = -INFINITY;
            for (int action: mdp.getAvailableActions(x)) {
                double r_opt = estimated_rewards[x][action] + reward_uncertainty[x][action];
                double p_opt = optimize(estimated_transition_chances[x][action], v, transition_chance_uncertainty[x][action]/2);
                double q = 0.5 * (r_opt + p_opt);
                
                if (q>max_q) {
                    // max_q =          max_{a \in A(x)} Q_{t+1}*(x, a)
                    // best_action[x] = argmax of above
                    max_q = q;
                    best_action[x] = action;
                };
            }
            w[x] = max_q;
        }
        
        float max_dv = -INFINITY;
        float min_dv = INFINITY;
        for (int x=0; x<n; x++) {
            float dv = w[x]-v[x];
            if (dv > max_dv)
                max_dv = dv;
            if (dv < min_dv)
                min_dv = dv;
            v[x] = w[x];
        }

        float v0 = v[0];
        for (int x=0; x<n; x++)
            v[x] -= v0;
        
        if (max_dv - min_dv < eps) {
            vector<int> pol;
            for (int x=0; x<n; x++)
                pol.push_back(best_action[x]);
            Policy policy = {{pol}};
            return policy;
        }
    }
    cout << "Extended value iteration did not end within " << max_steps << " steps" << endl;
    Policy policy = {{}};
    return policy;
}

vector<double> ucrl2(MDP &mdp, float delta, int max_steps) {
    int t=0;
    double total_rewards;
    vector<double> rewards_v;

    int states = mdp.getStates();
    int actions = mdp.getMaxAction();
    int state = mdp.getState();
    
    Matrix<int> visits_before_episode(states, vector<int>(actions, 0));
    Matrix<int> visits_during_episode(states, vector<int>(actions, 0));
    Matrix<float> observed_rewards_before_episode(states, vector<float>(actions, 0.0));
    Matrix<float> observed_rewards_during_episode(states, vector<float>(actions, 0.0));
    Matrix3D<int> observed_transitions_before_episode(states, Matrix<int>(actions, vector<int>(states, 0)));
    Matrix3D<int> observed_transitions_during_episode(states, Matrix<int>(actions, vector<int>(states, 0)));

    Matrix<double> estimated_rewards(states, vector<double>(actions, 0.0));
    Matrix<double> reward_uncertainty(states, vector<double>(actions, 0.0));
    Matrix3D<double> estimated_transition_chances(states, Matrix<double>(actions, vector<double>(states, 0)));
    Matrix<double> transition_chance_uncertainty(states, vector<double>(actions, 0.0));

    for (int k=1; k>0; k++) {
        int start = t;

        // Initialize state-action counts, accumulated rewards and transition counts for the current episode
        for (int x=0; x<states; x++) {
            for (int a: mdp.getAvailableActions(x)) {
                visits_before_episode[x][a] += visits_during_episode[x][a];
                visits_during_episode[x][a] = 0;
                observed_rewards_before_episode[x][a] += observed_rewards_during_episode[x][a];
                observed_rewards_during_episode[x][a] = 0.0;

                estimated_rewards[x][a] = observed_rewards_before_episode[x][a] / max(1, visits_before_episode[x][a]);
                for (int y=0; y<states; y++) {
                    observed_transitions_before_episode[x][a][y] += observed_transitions_during_episode[x][a][y];
                    observed_transitions_during_episode[x][a][y] = 0.0;
                    estimated_transition_chances[x][a][y] = (double) observed_transitions_before_episode[x][a][y] / max(1, visits_before_episode[x][a]);
                    if (visits_before_episode[x][a] == 0)
                        estimated_transition_chances[x][a][y] = 1.0/states;
                }

                reward_uncertainty[x][a] = sqrt(3.5 * log(2*states*actions*start/delta) / max(1, visits_before_episode[x][a]));
                transition_chance_uncertainty[x][a] = sqrt(14 * log(2*actions*start/delta) / max(1, visits_before_episode[x][a]));
            }
        }

        // Compute optimal policy for optimist MDP (EVI)
        Policy policy = extended_value_iteration(mdp, estimated_transition_chances, estimated_rewards, transition_chance_uncertainty, reward_uncertainty, 1000, 0.000001);
        Agent agent = Agent(mdp, policy);

        while (visits_during_episode[state][policy(state, 0)] < max(1, visits_before_episode[state][policy(state, 0)])) {
            float rewards;
            agent.usePolicy(rewards);
            
            int x = state;
            int a = policy(x, 0);
            int y = mdp.getState();

            visits_during_episode[x][a]++;
            observed_transitions_during_episode[x][a][y]++;
            observed_rewards_during_episode[x][a] += rewards;
            total_rewards += rewards;
            
            rewards_v.push_back(rewards);
            t++;
            if (t==max_steps)
                return rewards_v;
            state = y;
        }
    }
    return {};
}