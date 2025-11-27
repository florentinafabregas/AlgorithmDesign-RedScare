from collections import deque
from parser import GraphParser
from typing import Dict
import networkx as nx  
import csv
from pathlib import Path
import glob

# Utility functions

def reachable(graph, start) -> Dict[str, bool]:
    """
    Performs BFS from 'start' and returns a dict with all vertices
    that are reachable from it.
    """
    reach = {v: False for v in graph.vertices}
    q = deque([start])
    reach[start] = True
    while q:
        u = q.popleft()
        for v in graph.neighbors(u):
            if not reach[v]:
                reach[v] = True
                q.append(v)
    return reach

def bfs_forward(graph, start):
    '''
    Returns the set of vertices reachable from 'start'.
    '''
    return {v for v, r in reachable(graph, start).items() if r}

def bfs_reverse(graph, target):
    """
    Return the set of vertices that 'target' can reach
    on directed edges.
    """
    if not graph.directed:
        return bfs_forward(graph, target)
    rev_adj = {v: [] for v in graph.vertices}
    for u in graph.vertices:
        for v in graph.neighbors(u):
            rev_adj[v].append(u)
    visited = {v: False for v in graph.vertices}
    q = deque([target])
    visited[target] = True
    while q:
        u = q.popleft()
        for v in rev_adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    return {v for v, ok in visited.items() if ok}

def is_DAG(graph):
    '''
    Kahn's algorithm for detecting directed acyclic graphs.
    '''
    if not graph.directed:
        return False
    indeg = {u: 0 for u in graph.vertices}
    for u in graph.vertices:
        for v in graph.neighbors(u):
            indeg[v] += 1
    q = deque([u for u in graph.vertices if indeg[u] == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in graph.neighbors(u):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(graph.vertices)

def is_tree(graph):
    """
    Returns True if undirected graph is a tree.

    A graph is a tree if it is undirected, it has exactly
    V-1 edges, and if it is connected (all vertices are reachable
    from any vertex).
    """
    if graph.directed:
        return False

    V = len(graph.vertices)
    E = sum(len(graph.neighbors(u)) for u in graph.vertices) // 2
    if E != V - 1:
        return False

    reach = reachable(graph, next(iter(graph.vertices)))
    return all(reach.values())



# Solver functions

# If DAG >> solve bfs from s and from t that end in r

def some_DAG(graph):
    '''
    Returns True if a red vertex exists on a directed
    path from 'start' to 'target'. 
    
    Computes reachability from 'start' and reversed reachability
    from 'target', and checks for existance of red vertices
    on both sets.
    '''
    reach_s = reachable(graph, graph.s)
    reach_t = bfs_reverse(graph, graph.t)
    for r in graph.red:
        if reach_s.get(r, False) and r in reach_t:
            return True
    return False

# If tree >> unique path from s to t. Compute and check if red. 

def some_TREE(graph):
    '''
    Returns True if the unique s-t path contains a red vertex.
    Uses BFS to find the unique s-t path, and scan for red vertices.
    '''
    parent = {v: None for v in graph.vertices}
    visited = set([graph.s])
    q = deque([graph.s])

    while q:
        u = q.popleft()
        if u == graph.t:
            break
        for v in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                parent[v] = u
                q.append(v)

    if parent[graph.t] is None and graph.s != graph.t:
        return False  # no path

    path = []
    cur = graph.t
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()  

    for v in path:
        if v in graph.red:
            return True
    return False


# General algorithm for undirected
# EXPONENTIAL .. do not use

def _multi_source_reachable(graph, sources):
    """
    For an undirected graph: return dict v -> True/False saying
    whether v can reach at least one vertex in 'sources'.
    """
    reach = {v: False for v in sorted(graph.vertices)}
    q = deque()

    for s in sources:
        if s in reach and not reach[s]:
            reach[s] = True
            q.append(s)

    while q:
        u = q.popleft()
        for v in sorted(graph.neighbors(u)):
            if not reach[v]:
                reach[v] = True
                q.append(v)

    return reach


def some_undirected_pruned(graph, node_budget=400):
    """
    DFS over simple s-t paths with pruning:
      - prune vertices that cannot reach t (can_reach_t[v] == False)
      - if we haven't seen a red yet, prune vertices that cannot reach any red vertex
        (can_reach_red[v] == False)
      - stop immediately when we find an s-t path with at least one red vertex
      - abort if node_budget is exceeded (raise RuntimeError)
    """
    # Precompute pruning information (undirected, so plain BFS is enough)
    can_reach_t = _multi_source_reachable(graph, [graph.t])
    can_reach_red = _multi_source_reachable(graph, list(graph.red))

    visited = {v: False for v in sorted(graph.vertices)}
    expansions = 0
    found = False

    def dfs(v, got_red):
        nonlocal expansions, found

        if found:
            return

        if expansions >= node_budget:
            raise RuntimeError("node budget exhausted in some_undirected_pruned")
        expansions += 1

        # Prune if t is unreachable from here (even ignoring visited)
        if not can_reach_t[v]:
            return

        if not got_red and not can_reach_red[v]:
            return

        # Success condition - reached t having seen at least one red
        if v == graph.t and got_red:
            found = True
            return

        for u in sorted(graph.neighbors(v)):
            if not visited[u]:
                visited[u] = True
                dfs(u, got_red or (u in graph.red))
                visited[u] = False
                if found:
                    return

    visited[graph.s] = True
    dfs(graph.s, graph.s in graph.red)
    return found



# MAIN solver

def solve_some(graph):
    reach = reachable(graph, graph.s)
    if not reach.get(graph.t, False):
        return False, "unreachable"
    
    if len(graph.red) == 0:
        return False, "no-red"  
    
    if graph.s in graph.red or graph.t in graph.red or set(graph.red) == set(graph.vertices):
        return True, "trivial-red"

    if graph.directed:
        if is_DAG(graph):
            ans = some_DAG(graph)
            return ans, "dag"
        else:
            return "?!", "general directed" # NP-hard for directed-cyclic
    
    if is_tree(graph):
        ans = some_TREE(graph)
        return ans, 'tree'
    
    else:

        return '?!', "general undirected"

        # General undirected case: DFS + pruning + node budget
        # try:
        #     ans = some_undirected_pruned(graph, node_budget=400)
        #     return ans, "dfs-pruned"
        # except RuntimeError:
        #     # If the budget is exhausted, report unknown / hard instance.
        #     return "?!", "dfs-pruned-budget"
    


# Main loop with CSV output

if __name__ == "__main__":
    data_dir = "../red-scare/data"
    graph_files = sorted(glob.glob(f"{data_dir}/*.txt"))

    results = []
    for fp in graph_files:
        name = Path(fp).name
        try:
            g = GraphParser.from_file(fp)
            ans, detail = solve_some(g)
            results.append((name, ans, detail))
            print(f"{name}: {ans} ({detail})")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"{name}: ERROR — {e}")

    # write CSV
    csv_file = "../results/some_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "answer", "detail"])
        writer.writerows(results)
