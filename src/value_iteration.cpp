#include "mdp.cpp"

Policy value_iteration(OfflineMDP *mdp, int n, int max_steps, float eps, float *g) {
    /* 
        Runs value iteration on an MDP with n states until the span of the difference gets lower than eps
        Returns corresponding policy and sends corresponding gain to g
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
                float q = mdp->getRewards(x, a);
                for (int y=0; y<n; y++) q += mdp->getTransitionChance(x, a, y) * v[y];
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
        float v0 = v[0];
        for (int x=0; x<n; x++) v[x] -= v0;

        if (max_dv - min_dv < eps) {
            vector<int> pol;
            *g = (max_dv + min_dv)/2;
            for (int x=0; x<n; x++) pol.push_back(best_action[x]);
            Policy policy = {{pol}};
            return policy;
        }
    }
    Policy policy = {{}};
    return policy;
}

Policy value_iteration(OfflineMDP *mdp, int n, int max_steps, float eps) {
    float g;
    return (value_iteration(mdp, n, max_steps, eps, &g));
}

vector<float> invariant_measure(OfflineMDP *mdp, Policy *policy) {
    /* Get invariant measure of a policy with value iteration */
    
    vector<float> ans;

    int n = mdp->getStates();
    for (int x=0; x<n; x++) {
        vector<int> actions[n];
        int max_action = 0;
        for (int i=0; i<n; i++) {
            int action = (*policy)(i, 0);
            actions[i] = {action};
            if (action > max_action) max_action = action;
        }

        float **rewards = new float*[n];
        for (int i=0; i<n; i++) {
            rewards[i] = new float[max_action];
            for (int a=0; a<max_action; a++) rewards[i][a] = 0.0;
        }
        rewards[x][(*policy)(x, 0)] = 1.0;

        float g;
        OfflineMDP nmdp = OfflineMDP(n, actions, mdp->getTransitionKernel(), rewards);
        value_iteration(&nmdp, n, 1e6, 1e-6, &g);
        ans.push_back(g);
    }
    return ans;
}