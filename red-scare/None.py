import os
import re
import time
import csv
from collections import defaultdict, deque

# ---------------------------
# Parsing utilities
# ---------------------------

VERTEX_RE = re.compile(r"^([_a-z0-9]+)(\s*\*)?\s*$")
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
        self.is_red = []        # list[bool] indexed by id
        self.adj = defaultdict(list)  # directed adjacency (u -> list of v)

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

    # vertices
    for _ in range(n):
        line = next(it).strip()
        mobj = VERTEX_RE.match(line)
        if not mobj:
            raise ValueError(f"Bad vertex line: {line}")
        name, star = mobj.group(1), mobj.group(2)
        g.add_vertex(name, bool(star))

    if s_name not in g.name_to_id or t_name not in g.name_to_id:
        raise ValueError("s or t not in vertex list")
    g.s = g.name_to_id[s_name]
    g.t = g.name_to_id[t_name]

    # edges
    for _ in range(m):
        line = next(it).strip()
        md = EDGE_DIR_RE.match(line)
        mu = EDGE_UND_RE.match(line) if md is None else None
        if md:
            u, v = md.group(1), md.group(2)
            g.add_arc(g.name_to_id[u], g.name_to_id[v])
        elif mu:
            u, v = mu.group(1), mu.group(2)
            ui, vi = g.name_to_id[u], g.name_to_id[v]
            g.add_arc(ui, vi)
            g.add_arc(vi, ui)
        else:
            raise ValueError(f"Bad edge line: {line}")

    return g


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


# ---------------------------
# NONE algorithm
# ---------------------------

def none_length(g):
    """
    Return length of a shortest s–t path whose *internal* vertices are not red.
    s and/or t may be red. Works for directed or undirected graphs.
    """
    if g.s == g.t:
        return 0

    def allowed(v):
        # internal vertices cannot be red
        return (not g.is_red[v]) or v == g.s or v == g.t

    if not allowed(g.s) or not allowed(g.t):
        return -1

    INF = 10**15
    dist = [INF] * g.n
    dist[g.s] = 0
    q = deque([g.s])

    while q:
        u = q.popleft()
        for v in g.adj[u]:
            if not allowed(v):
                continue
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                if v == g.t:
                    return dist[v]
                q.append(v)
    return -1


# ---------------------------
# Main: batch runner
# ---------------------------

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUT_CSV = os.path.join(BASE_DIR, "none_results.csv")

    graphs = load_all_graphs(DATA_DIR)
    print(f"Loaded {len(graphs)} graphs from {DATA_DIR}")

    rows = []
    for name, g in graphs.items():
        t0 = time.perf_counter()
        try:
            ans = none_length(g)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            rows.append({
                "instance": name,
                "n": g.n,
                "m": g.m,
                "r": g.r,
                "s": g.id_to_name[g.s],
                "t": g.id_to_name[g.t],
                "answer": ans,
                "time_ms": elapsed_ms
            })
            print(f"{name:30s}  ans={ans:3}  {elapsed_ms:5d} ms")
        except Exception as e:
            print(f"{name:30s}  ERROR: {e}")

    # write results
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance","n","m","r","s","t","answer","time_ms"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"\nWrote results to {OUT_CSV}")
