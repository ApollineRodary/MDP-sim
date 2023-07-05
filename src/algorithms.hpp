#include "mdp.hpp"

Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps, float &g);
Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps);
vector<float> invariant_measure(OfflineMDP &mdp, Policy &policy);
vector<float> invariant_measure_estimate(Agent &agent, int steps);

vector<double> ucrl2(MDP &mdp, float delta, int max_steps);