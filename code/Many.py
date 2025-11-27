import os
import sys
import re
from collections import defaultdict, deque, Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "red-scare", "data"))

def load_all_graphs(data_dir):
    graphs = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            g = parse_graph_from_lines(lines)
            graphs[os.path.splitext(fname)[0]] = g
    return graphs

VERTEX_RE = re.compile(r"^([_a-z0-9]+)(\s*\*)?$")
EDGE_DIR_RE = re.compile(r"^([_a-z0-9]+)\s*->\s*([_a-z0-9]+)$")
EDGE_UND_RE = re.compile(r"^([_a-z0-9]+)\s*--\s*([_a-z0-9]+)$")

class Graph:
    def __init__(self):
        self.n = 0
        self.m = 0
        self.r = 0
        self.s = None
        self.t = None
        self.name_to_id = {}
        self.id_to_name = []
        self.is_red = []     
        self.adj = defaultdict(list)

    def add_vertex(self, name, red):
        vid = len(self.id_to_name)
        self.name_to_id[name] = vid
        self.id_to_name.append(name)
        self.is_red.append(bool(red))
        return vid

    def add_arc(self, u, v):
        self.adj[u].append(v)

    def V(self):
        return range(len(self.id_to_name))

def parse_graph_from_lines(lines):
    it = iter(lines)
    n, m, r = map(int, next(it).strip().split())
    s_name, t_name = next(it).strip().split()
    g = Graph()
    g.n, g.m, g.r = n, m, r

    names = []
    reds = []
    for _ in range(n):
        line = next(it).strip()
        mobj = VERTEX_RE.match(line)
        if not mobj:
            raise ValueError(f"Bad vertex line: {line}")
        name, star = mobj.group(1), mobj.group(2)
        names.append(name)
        reds.append(bool(star))
    for name, red in zip(names, reds):
        g.add_vertex(name, red)

    if s_name not in g.name_to_id or t_name not in g.name_to_id:
        raise ValueError("s or t not in vertex list")
    g.s = g.name_to_id[s_name]
    g.t = g.name_to_id[t_name]

    for _ in range(m):
        line = next(it).strip()
        if not line:
            raise ValueError("Missing edge line")
        md = EDGE_DIR_RE.match(line)
        mu = EDGE_UND_RE.match(line) if md is None else None
        if md:
            u, v = md.group(1), md.group(2)
            if u not in g.name_to_id or v not in g.name_to_id:
                raise ValueError(f"Unknown vertex in edge: {line}")
            g.add_arc(g.name_to_id[u], g.name_to_id[v])
        elif mu:
            u, v = mu.group(1), mu.group(2)
            if u not in g.name_to_id or v not in g.name_to_id:
                raise ValueError(f"Unknown vertex in edge: {line}")
            ui, vi = g.name_to_id[u], g.name_to_id[v]
            g.add_arc(ui, vi)
            g.add_arc(vi, ui)
        else:
            raise ValueError(f"Bad edge line: {line}")

    return g

def reachable(g, src):
    q = deque([src])
    seen = [False]*g.n
    seen[src] = True
    while q:
        u = q.popleft()
        for v in g.adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen

def is_dag(g):
    # Kahn's algorithm on directed view
    indeg = [0]*g.n
    for u in g.V():
        for v in g.adj[u]:
            indeg[v] += 1
    q = deque([u for u in g.V() if indeg[u] == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in g.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == g.n

def undirected_only(g):
    # If every arc has its reverse, we consider it undirected-only.
    S = set()
    for u in g.V():
        for v in g.adj[u]:
            S.add((u, v))
    for (u, v) in S:
        if (v, u) not in S:
            return False
    return True

def is_tree_undirected(g):
    if not undirected_only(g):
        return False
    # Count undirected edges once: |E| = half of arcs
    arc_count = sum(len(g.adj[u]) for u in g.V())
    E = arc_count // 2
    seen = reachable(g, g.s)
    if sum(seen) != g.n:
        return False
    return E == g.n - 1

# Exact: DAG DP (longest red-weighted s->t)
def many_dag(g):
    # Topological order with Kahn
    indeg = [0]*g.n
    for u in g.V():
        for v in g.adj[u]:
            indeg[v] += 1
    q = deque([u for u in g.V() if indeg[u] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in g.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != g.n:
        raise ValueError("Graph is not a DAG")

    NEG_INF = -10**15
    dist = [NEG_INF]*g.n
    dist[g.s] = 1 if g.is_red[g.s] else 0
    for u in topo:
        if dist[u] == NEG_INF:
            continue
        for v in g.adj[u]:
            cand = dist[u] + (1 if g.is_red[v] else 0)
            if cand > dist[v]:
                dist[v] = cand

    return -1 if dist[g.t] == NEG_INF else dist[g.t]


# Exact: Undirected tree (unique path)
def many_undirected_tree(g):
    # Recover the unique path with BFS parents
    parent = [-1]*g.n
    q = deque([g.s])
    parent[g.s] = g.s
    while q:
        u = q.popleft()
        if u == g.t:
            break
        for v in g.adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    if parent[g.t] == -1:
        return -1

    path = []
    cur = g.t
    while cur != g.s:
        path.append(cur)
        cur = parent[cur]
    path.append(g.s)
    path.reverse()
    return sum(1 for v in path if g.is_red[v])


# General fallback: DFS with pruning

def many_general_fallback(g, node_budget=50000):
    if g.s == g.t:
        return 1 if g.is_red[g.s] else 0
    
    rev_adj = defaultdict(list)
    for u in g.V():
        for v in g.adj[u]:
            rev_adj[v].append(u)
    can_reach_t = [False] * g.n
    dq = deque([g.t])
    can_reach_t[g.t] = True
    while dq:
        x = dq.popleft()
        for p in rev_adj[x]:
            if not can_reach_t[p]:
                can_reach_t[p] = True
                dq.append(p)

    total_red = sum(1 for v in g.V() if g.is_red[v])

    best = -1
    visited = [False] * g.n
    expansions = 0
    budget_exhausted = False  # new flag

    # very loose optimistic bound: current_red + (total_red - red_seen_global)
    red_seen_global = set()

    def dfs(u, cur_red):
        nonlocal best, expansions, budget_exhausted
        if budget_exhausted:
            return
        if expansions >= node_budget:
            budget_exhausted = True
            return
        expansions += 1

        # prune if t not reachable from u
        if not can_reach_t[u]:
            return

        if u == g.t:
            if cur_red > best:
                best = cur_red
            return

        # optimistic bound (loose but cheap)
        optimistic = cur_red + (total_red - len(red_seen_global))
        if optimistic <= best:
            return

        for v in g.adj[u]:
            if budget_exhausted:
                return
            if not visited[v]:
                visited[v] = True
                added = 0
                if g.is_red[v] and v not in red_seen_global:
                    red_seen_global.add(v)
                    added = 1
                dfs(v, cur_red + added)
                if added == 1:
                    red_seen_global.remove(v)
                visited[v] = False

    visited[g.s] = True
    start_red = 1 if g.is_red[g.s] else 0
    if g.is_red[g.s]:
        red_seen_global.add(g.s)
    dfs(g.s, start_red)

    # If we ran out of budget - mark instance unsolved
    if budget_exhausted:
        raise RuntimeError("node budget exhausted in many_general_fallback")

    return best


def solve_many(g):
    reach = reachable(g, g.s)
    if not reach[g.t]:
        return -1

    if is_dag(g):
        return many_dag(g)

    if undirected_only(g) and is_tree_undirected(g):
        return many_undirected_tree(g)

    return many_general_fallback(g, node_budget=50000)

def solve_many_with_tag(g):
    reach = reachable(g, g.s)
    if not reach[g.t]:
        return -1, "unreachable"

    if is_dag(g):
        return many_dag(g), "dag"

    if undirected_only(g) and is_tree_undirected(g):
        return many_undirected_tree(g), "tree"

    # fallback
    val = many_general_fallback(g, node_budget=50000)
    return val, "fallback"


def main():
    data = sys.stdin.read().strip().splitlines()
    g = parse_graph_from_lines(data)
    ans = solve_many(g)
    print(ans)


if __name__ == "__main__":
    import csv
    import time

    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    OUT_CSV = os.path.join(RESULTS_DIR, "many_results.csv")

    graphs = load_all_graphs(DATA_DIR)
    print(f"Loaded {len(graphs)} graphs from {DATA_DIR}")

    rows = []
    solved = 0
    for name, g in graphs.items():
        t0 = time.perf_counter()
        try:
            val, tag = solve_many_with_tag(g)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            rows.append({
                "instance": name,
                "n": g.n,
                "m": g.m,
                "r": g.r,
                "s": g.id_to_name[g.s],
                "t": g.id_to_name[g.t],
                "answer": val,
                "solver": tag,
                "time_ms": elapsed_ms,
            })
            if val != -1:
                solved += 1
            print(f"{name:30s}  ans={val:3}  solver={tag:9s}  {elapsed_ms:5d} ms")
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            rows.append({
                "instance": name,
                "n": g.n, "m": g.m, "r": g.r,
                "s": g.id_to_name[g.s], "t": g.id_to_name[g.t],
                "answer": "?!",
                "solver": "?!",
                "time_ms": elapsed_ms,
                "error": str(e),
            })
            print(f"{name:30s}  ERROR: {e}")

    # Only keep instance, n, answer
    fieldnames = ["instance", "n", "answer"]

    OUT_CSV = os.path.join(RESULTS_DIR, "many_results.csv")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({
                "instance": row["instance"],
                "n": row["n"],
                "answer": row["answer"],
            })

    print(f"\nWrote results to {OUT_CSV}")
    print(f"Solved (answer != -1): {solved}/{len(graphs)}")

    # Summary of 'answer' column
    answer_counts = Counter(row["answer"] for row in rows)

    print("\nSummary of 'answer' column:")
    for val, cnt in answer_counts.most_common():
        print(f"  {val}: {cnt}")