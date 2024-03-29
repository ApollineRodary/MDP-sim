#include <stdexcept>
#include <algorithm>
#include "algorithms.hpp"
#include "io.hpp"
#include <iostream>
#include <iomanip>

tuple<Policy, double, vector<double>> value_iteration(OfflineMDP &mdp, int max_steps, float eps) {
    /* 
        Runs value iteration on an MDP with n states until the span of the difference gets lower than eps
        Returns the corresponding policy, the gain and the bias
    */

    if (eps<=0)
        throw invalid_argument("eps must be a positive value");

    int n = mdp.getStates();
    int a = mdp.getMaxAction();

    vector<double> v(n, 0.0);
    vector<double> w(n);
    vector<int> best_action(n);

    for (int t=0;; t++) {
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

        double max_dv = -INFINITY;
        double min_dv = INFINITY;
        for (int x=0; x<n; x++) {
            double dv = w[x]-v[x];
            if (dv > max_dv)
                max_dv = dv;
            if (dv < min_dv)
                min_dv = dv;
            v[x] = w[x];
        }
        double v0 = v[0];
        for (int x=0; x<n; x++)
            v[x] -= v0;
        
        double span = max_dv-min_dv;
        if (span<eps || t==max_steps) {
            vector<int> pol;
            double g = (max_dv + min_dv)/2;
            for (int x=0; x<n; x++)
                pol.push_back(best_action[x]);
            Policy policy = {{pol}};
            return tuple(policy, g, v);
        }
    }
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
        
        OfflineMDP nmdp = OfflineMDP(actions, mdp.getTransitionKernel(), rewards);
        double g = get<1>(value_iteration(nmdp, 1e5, 1e-5));
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
    
    vector<float> frequency(agent.getMDP().getStates(), 0.0);

    for (int i=0; i<steps; i++) {
        agent.usePolicy();
        frequency[agent.getMDP().getState()]++;
    }
    
    vector<float> d;
    for (int f: frequency)
        d.push_back(((float) f)/steps);
    return d;
}

double gap_regret(int x, int a, OfflineMDP &mdp) {
    auto vi_data = value_iteration(mdp, 1e5, 1e-5);
    double g = get<1>(vi_data);
    vector<double> h = get<2>(vi_data);
    
    double reward_gap = g - mdp.getRewards(x, a);
    double bias_gap = h[x];
    for (int y=0; y<mdp.getStates(); y++)
        bias_gap -= mdp.getTransitionChance(x, a, y)*h[y];

    return reward_gap + bias_gap;
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
    
    // Create q similar to p
    vector<double> q(n);
    for (int i=0; i<n; i++)
        q[s[i]] = p[s[i]];
    
    // Add as much weight as possible to q_i for i maximizing u_i, taking from q_j for j minimizing u_j
    int i=0, j=n-1;
    while (i<j) {
        double m = min({0.5*eps, 1.0-q[s[i]], q[s[j]]});

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

    for (int i=0; i<n; i++)
        q[i] = round(q[i]*1e5) / 1e5;

    return inner_product(q.begin(), q.end(), u.begin(), 0.0);
}

tuple<Policy, double, vector<double>> extended_value_iteration(MDP &mdp, ExtendedMDP &extended_mdp, int max_steps, float eps) {
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
    
    double g;
    for (int t=0;; t++) {
        for (int x=0; x<n; x++) {
            float max_q = -INFINITY;
            for (int action: mdp.getAvailableActions(x)) {
                double r_opt = extended_mdp.getOptimistReward(x, action);
                double p_opt = optimize(extended_mdp.estimated_transition_chances[x][action], v, extended_mdp.transition_chance_uncertainty[x][action]);
                double q = r_opt + p_opt;
                
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
        
        float span = max_dv - min_dv;
        if (span < eps || t > max_steps) {
            g = (max_dv + min_dv) / 2;
            break;
        }
    }

    vector<int> pol;
    for (int x=0; x<n; x++)
        pol.push_back(best_action[x]);
    Policy policy = {{pol}};

    return tuple(policy, g, v);
}

pair<History, EpisodeHistory> ucrl2(MDP &mdp, float delta, int steps, int episodes, const History &context) {
    /*
        Plays UCRL2 on MDP mdp for a given duration, given the previous history provided by context
        Returns observed history and vector of episode start times
    */
    
    int t = context.size() + 1;
    double total_rewards;

    History history(0);
    EpisodeHistory episode_history;

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
    ExtendedMDP extended_mdp(estimated_rewards, reward_uncertainty, estimated_transition_chances, transition_chance_uncertainty);

    // Read previous history
    int x=state, y=state, a;
    double r;
    for (Event e: context) {
        x = get<0>(e);
        a = get<1>(e);
        y = get<2>(e);
        r = get<3>(e);

        visits_during_episode[x][a]++;
        observed_rewards_during_episode[x][a] += r;
        observed_transitions_during_episode[x][a][y]++;
    }
    state = y;

    // Start UCRL2
    int k=0;
    while (true) {
        k++;
        int start = t;

        // Initialize state-action counts, accumulated rewards and transition counts for the current episode
        for (int x=0; x<states; x++) {
            for (int a: mdp.getAvailableActions(x)) {
                visits_before_episode[x][a] += visits_during_episode[x][a];
                visits_during_episode[x][a] = 0;
                observed_rewards_before_episode[x][a] += observed_rewards_during_episode[x][a];
                observed_rewards_during_episode[x][a] = 0.0;

                for (int y=0; y<states; y++) {
                    observed_transitions_before_episode[x][a][y] += observed_transitions_during_episode[x][a][y];
                    observed_transitions_during_episode[x][a][y] = 0.0;
                }
            }
        }
        extended_mdp.update(mdp, visits_before_episode, observed_rewards_before_episode, observed_transitions_before_episode, start, delta);

        // Compute optimal policy for optimist MDP (EVI)
        auto evi_output = extended_value_iteration(mdp, extended_mdp, 1000, 1.0/sqrt(start));
        Policy policy = get<0>(evi_output);
        Agent agent = Agent(mdp, policy);
        episode_history.push_back(pair(start, policy));

        // Iterate episode until a state-action pair has been visited in the current episode as many times as all episodes prior
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
            
            Event event(x, a, y, rewards);
            history.push_back(event);

            t++;
            if (steps>0)
                show_loading_bar("Running UCRL2...   ", t, steps);
            state = y;

            if (t==steps)
                break;
        }

        if (t==steps || k==episodes)
            break;
    }

    return pair(history, episode_history);
}

bool compare_policies(Policy &a, Policy &b, int states) {
    for (int x=0; x<states; x++)
        if (a(x, 0) != b(x, 0))
            return false;
    return true;
}

int find_bad_episode(History &history, EpisodeHistory &episode_history, Policy &opt_policy, int min) {
    /** Finds index of a bad episode late enough in a UCRL2 run, 0 if no such episode can be found
      * . history: plays the recorded UCRL2 run
      * . episode_history: episodes of the recorded UCRL2 run
      * . opt_policy: the optimal policy - a "bad" episode is an episode that uses a policy different than this one
      * . min: the minimum starting time of the returned episode
      */
    
    int k=-1;
    for (auto episode: episode_history) {
        k++;
        int start_time = episode.first;
        Policy policy = episode.second;

        if (start_time<min)
            continue;
        
        if (compare_policies(policy, opt_policy, policy.v.size()))
            continue;
        
        show_policy(policy);
        return k;
    }
    return 0;
}

pair<vector<double>, vector<double>> performance_test(OfflineMDP &mdp, Policy &policy, History &past, History &history, int start, int duration, double delta) {
    /** Compares optimistic gain under the given policy throughout the provided history, and optimistic value without the policy restraint
      * . mdp: the MDP to run EVI on
      * . policy: the policy that is being evaluated
      * . past: the history of UCRL2 plays before the recorded episode
      * . history: the history of UCRL2 plays of the episode to evaluate
      * . start: when the episode started
      * . delta: the parameter for computing confidence intervals
      */

    int n = mdp.getStates();
    int actions = mdp.getMaxAction();

    Matrix<double> estimated_rewards(n, vector<double>(actions, 0.0));
    Matrix<double> reward_uncertainty(n, vector<double>(actions, 0.0));
    Matrix3D<double> estimated_transition_chances(n, Matrix<double>(actions, vector<double>(n, 0)));
    Matrix<double> transition_chance_uncertainty(n, vector<double>(actions, 0.0));
    ExtendedMDP extended_mdp(estimated_rewards, reward_uncertainty, estimated_transition_chances, transition_chance_uncertainty);

    Matrix<int> visits(n, vector<int>(actions, 0));
    Matrix<float> observed_rewards(n, vector<float>(actions, 0.0));
    Matrix3D<int> observed_transitions(n, Matrix<int>(actions, vector<int>(n, 0)));

    Matrix<int> policy_actions(n);
    for (int x=0; x<n; x++)
        policy_actions[x] = {policy(x, 0)};
    MDP mdp_with_policy_actions(policy_actions, mdp.getTransitionKernel(), mdp.getRewardMatrix());

    vector<double> g_opt(duration);
    vector<double> g(duration);

    int x, a, y;
    double r;

    for (Event e: past) {
        x = get<0>(e);
        a = get<1>(e);
        y = get<2>(e);
        r = get<3>(e);

        visits[x][a]++;
        observed_rewards[x][a] += r;
        observed_transitions[x][a][y]++;
    }

    int t=start;
    for (Event e: history) {
        if (t >= start+duration)
            break;

        x = get<0>(e);
        a = get<1>(e);
        y = get<2>(e);
        r = get<3>(e);

        visits[x][a]++;
        observed_rewards[x][a] += r;
        observed_transitions[x][a][y]++;

        extended_mdp.update(mdp, visits, observed_rewards, observed_transitions, t, delta);

        auto evi_output_opt = extended_value_iteration(mdp, extended_mdp, 1e3, 1e-5);
        g_opt[t-start] = get<1>(evi_output_opt);

        auto evi_output = extended_value_iteration(mdp_with_policy_actions, extended_mdp, 1e3, 1e-5);
        g[t-start] = get<1>(evi_output);

        t++;
    }

    return pair(g, g_opt);
}