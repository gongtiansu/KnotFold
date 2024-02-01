#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace mcf {
template <class T> struct mcmf {
  struct edge {
    int dest, back;
    T cap, cost;
  };
  std::vector<std::vector<edge>> g;
  T total_flow, total_cost;

  explicit mcmf(int n) : g(n), total_flow(0), total_cost(0) {}

  void add_edge(int u, int v, T cap, T cost) {
    edge a{v, (int)g[v].size(), cap, cost}; // NOLINT
    edge b{u, (int)g[u].size(), 0, -cost};  // NOLINT
    g[u].push_back(a);
    g[v].push_back(b);
  }

  bool spfa_flow(int st, int ed) {
    int n = g.size();
    std::vector<T> dis(n, std::numeric_limits<T>::max() / 3);
    std::vector<bool> vis(n, false);
    std::vector<edge *> path(n, nullptr);
    dis[st] = 0, vis[st] = true;
    std::queue<int> q;
    q.push(st);
    while (!q.empty()) {
      int u = q.front();
      q.pop(), vis[u] = false;
      for (auto &forward : g[u]) {
        if (forward.cap && dis[forward.dest] > dis[u] + forward.cost + 1e-12) {
          dis[forward.dest] = dis[u] + forward.cost;
          path[forward.dest] = &forward;
          if (!vis[forward.dest]) {
            vis[forward.dest] = true, q.push(forward.dest);
          }
        }
      }
    }
    T flow = -1;
    bool found = false;
    if (path[ed] != nullptr && dis[ed] < 0) {
      found = true;
      for (int u = ed; u != st;) {
        flow = (flow == -1 ? path[u]->cap : std::min(flow, path[u]->cap));
        edge &back = g[path[u]->dest][path[u]->back];
        u = back.dest;
      }
      assert(flow == 1);
      total_flow += flow;
      for (int u = ed; u != st;) {
        edge &back = g[path[u]->dest][path[u]->back];
        path[u]->cap -= flow;
        back.cap += flow;
        total_cost += flow * path[u]->cost;
        u = back.dest;
      }
    }
    return found;
  }

  std::pair<T, T> run(int st, int ed) {
    while (spfa_flow(st, ed)) {
    }
    return {total_flow, total_cost};
  }
};
} // namespace min-cost flow

std::vector<std::vector<double>> parse_mat(const char *path) {
  auto fin = std::ifstream(path);
  std::string line;
  std::vector<std::vector<double>> a;
  while (std::getline(fin, line)) {
    std::stringstream ss(line);
    std::vector<double> b;
    double x;
    while (ss >> x) {
      b.push_back(x);
    }
    a.push_back(b);
  }
  return a;
}

int main(int argc, char *argv[]) {
  auto fg = parse_mat(argv[1]);
  auto bg = parse_mat(argv[2]);
  assert(fg.size() == bg.size());

  double lambda = 4.2;//set lambda
  
  const double eps = 1e-8;
  int n = fg.size();
  mcf::mcmf<long double> mcmf(2 * n + 2);
  int st = 0, ed = 2 * n + 1;
  for (int i = 1; i <= n; i++) {
    mcmf.add_edge(st, i, 1, 0);
    mcmf.add_edge(i + n, ed, 1, 0);
    for (int j = i + 3; j <= n; j++) {
      double p = log(fg[i - 1][j - 1] + eps) - log(bg[i - 1][j - 1] + eps);
      p -= (log(1 - fg[i - 1][j - 1] + eps) - log(1 - bg[i - 1][j - 1] + eps));
      p -= lambda;
      mcmf.add_edge(i, j + n, 1, -p);
      mcmf.add_edge(j, i + n, 1, -p);
    }
  }
  auto ret = mcmf.run(st, ed);

  for (int i = 1; i <= n; i++) {
    for (auto &edge : mcmf.g[i]) {
      if (edge.cap == 0 && edge.dest - n > i && edge.dest != ed) {
        std::cout << i << " " << edge.dest - n << "\n";
      }
    }
  }
  return 0;
}
