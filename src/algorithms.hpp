#include "mdp.hpp"

using Event = tuple<int, int, int, double>;
using History = vector<Event>;
using EpisodeHistory = vector<pair<int, Policy>>;

tuple<Policy, double, vector<double>> value_iteration(OfflineMDP &mdp, int max_steps, float eps);
vector<float> invariant_measure(OfflineMDP &mdp, Policy &policy);
vector<float> invariant_measure_estimate(Agent &agent, int steps);
double gap_regret(int x, int a, OfflineMDP &mdp);
pair<History, EpisodeHistory> ucrl2(MDP &mdp, float delta, int steps, int episodes = 0, const History &context = History(0));
int find_bad_episode(History &history, EpisodeHistory &episode_history, Policy &opt_policy, int min);
pair<vector<double>, vector<double>> performance_test(OfflineMDP &mdp, Policy &policy, History &past, History &history, int start, int duration, double delta);