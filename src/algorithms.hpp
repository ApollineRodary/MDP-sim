#include "mdp.hpp"

using Event = tuple<int, int, double>;
using History = vector<Event>;

Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps, float &g);
Policy value_iteration(OfflineMDP &mdp, int max_steps, float eps);
vector<float> invariant_measure(OfflineMDP &mdp, Policy &policy);
vector<float> invariant_measure_estimate(Agent &agent, int steps);

History ucrl2(MDP &mdp, float delta, int max_steps);