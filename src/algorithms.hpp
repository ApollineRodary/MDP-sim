#include "mdp.hpp"

using Event = tuple<int, int, double>;
using History = vector<Event>;

tuple<Policy, double, vector<double>> value_iteration(OfflineMDP &mdp, int max_steps, float eps);
vector<float> invariant_measure(OfflineMDP &mdp, Policy &policy);
vector<float> invariant_measure_estimate(Agent &agent, int steps);
double gap_regret(int x, int a, OfflineMDP &mdp);
History ucrl2(MDP &mdp, float delta, int max_steps);