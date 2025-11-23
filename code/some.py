from __future__ import annotations
from collections import deque
from parser import GraphParser
from typing import Dict
import networkx as nx  
import csv
from pathlib import Path
import glob

# Utility functions

def reachable(graph, start) -> Dict[str, bool]:
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
    return {v for v, r in reachable(graph, start).items() if r}

def bfs_reverse(graph, target):
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



# Solver functions

# If DAG >> solve bfs from s and from t that end in r

def some_DAG(graph):
    reach_s = reachable(graph, graph.s)
    reach_t = bfs_reverse(graph, graph.t)
    for r in graph.red:
        if reach_s.get(r, False) and r in reach_t:
            return True
    return False


def some_undirected(graph):  # Using max-flow (ERRORS - dont use)
    try:
        G = nx.DiGraph()
        # vertex splitting: v_in -> v_out
        for v in graph.vertices:
            cap = 1
            if v == graph.s or v == graph.t:
                cap = float('inf')
            G.add_edge(f"{v}_in", f"{v}_out", capacity=cap)

        # add edges
        for u in graph.adj:
            for v in graph.neighbors(u):
                if u <= v:  # avoid adding twice
                    G.add_edge(f"{u}_out", f"{v}_in", capacity=float('inf'))
                    G.add_edge(f"{v}_out", f"{u}_in", capacity=float('inf'))

        for r in graph.red:
            G[f"{r}_in"][f"{r}_out"]['capacity'] = 2
            flow_value, _ = nx.maximum_flow(G, f"{graph.s}_out", f"{graph.t}_in")
            if flow_value >= 1:
                return True
            G[f"{r}_in"][f"{r}_out"]['capacity'] = 1  # reset
        return False

    except Exception:
        return "?!"


def some_undirected_maxflow(graph):
    """
    Polynomial time check using vertex-disjoint paths.
    
    For each red vertex r, we check if there exist TWO vertex-disjoint paths:
    - Path 1: from s to r
    - Path 2: from r to t
    
    We do this by:
    1. Finding a flow from s to r
    2. Extracting vertices used
    3. Checking if flow exists from r to t avoiding those vertices
    """
    import networkx as nx
    
    for red_v in graph.red:
        # Special case
        if red_v == graph.s or red_v == graph.t:
            # Just check if any path exists
            try:
                if nx.has_path(nx.Graph([(u, v) for u in graph.vertices for v in graph.neighbors(u)]), 
                               graph.s, graph.t):
                    return True
            except:
                pass
            continue
        
        # Convert to NetworkX graph for easier manipulation
        G_nx = nx.Graph()
        for u in graph.vertices:
            for v in graph.neighbors(u):
                G_nx.add_edge(u, v)
        
        # Try to find vertex-disjoint paths s -> red_v and red_v -> t
        try:
            # Use NetworkX's built-in vertex disjoint paths finder
            # But it needs node_disjoint_paths which checks s->t, not s->r->t
            
            # So let's check differently:
            # Find ANY path s -> red_v
            path1 = nx.shortest_path(G_nx, graph.s, red_v)
            
            # Remove vertices from path1 (except red_v)
            G_reduced = G_nx.copy()
            for v in path1:
                if v != red_v and v != graph.t:
                    G_reduced.remove_node(v)
            
            # Check if path exists from red_v to t in reduced graph
            if nx.has_path(G_reduced, red_v, graph.t):
                return True
                
        except nx.NetworkXNoPath:
            continue
        except:
            continue
    
    return False
# -----------------

# General algorithm for undirected
# STILL EXPONENTIAL .. do not use ! 
def some_general(graph, budget=2000):
    """
    Backtracking solver for undirected (and small directed) graphs.
    
    Uses DFS with path tracking to find simple paths s -> red -> t.
    
    Complexity:
    - Worst case: Exponential O(V!)
    - Practice: Works well on sparse graphs, small graphs
    - Budget limit prevents exponential blowup
    
    Returns:
    - True if path through red exists
    - False if no path found within budget
    """
    # Try each red vertex as intermediate
    for red_v in graph.red:
        if _path_through_vertex_exists(graph, red_v, budget):
            return True
    return False


def _path_through_vertex_exists(graph, red_v, budget):
    """Check if simple path s -> red_v -> t exists."""
    
    # Special case: red is s or t
    if red_v == graph.s or red_v == graph.t:
        return _has_simple_path_dfs(graph, graph.s, graph.t, set(), budget)
    
    # Phase 1: Find simple path s -> red_v
    path_to_red = _find_simple_path_dfs(graph, graph.s, red_v, set(), budget // 2)
    if path_to_red is None:
        return False
    
    # Phase 2: Find simple path red_v -> t, avoiding vertices from phase 1
    used_vertices = set(path_to_red) - {red_v}
    path_from_red = _find_simple_path_dfs(graph, red_v, graph.t, used_vertices, budget // 2)
    
    return path_from_red is not None


def _has_simple_path_dfs(graph, start, target, forbidden, budget):
    """Check if simple path exists from start to target using DFS."""
    if start == target:
        return True
    if start in forbidden:
        return False
    
    visited = [False] * len(graph.vertices)
    vertex_to_idx = {v: i for i, v in enumerate(graph.vertices)}
    
    visited[vertex_to_idx[start]] = True
    stack = [(start, visited[:], 0)]
    
    nodes_explored = 0
    
    while stack and nodes_explored < budget:
        u, path_visited, _ = stack.pop()
        nodes_explored += 1
        
        if u == target:
            return True
        
        for v in graph.neighbors(u):
            v_idx = vertex_to_idx[v]
            if not path_visited[v_idx] and v not in forbidden:
                new_visited = path_visited[:]
                new_visited[v_idx] = True
                stack.append((v, new_visited, 0))
    
    return False


def _find_simple_path_dfs(graph, start, target, forbidden, budget):
    """
    Find a simple path from start to target using DFS.
    
    Args:
        graph: Graph object
        start: Starting vertex
        target: Target vertex
        forbidden: Set of vertices that cannot be used
        budget: Maximum nodes to explore
    
    Returns:
        List of vertices representing the path, or None if no path exists
    """
    if start == target:
        return [start]
    if start in forbidden:
        return None
    
    # Create vertex index mapping for efficient visited tracking
    vertex_to_idx = {v: i for i, v in enumerate(graph.vertices)}
    visited = [False] * len(graph.vertices)
    visited[vertex_to_idx[start]] = True
    
    # Stack: (current_vertex, path_so_far, visited_array)
    stack = [(start, [start], visited[:])]
    
    nodes_explored = 0
    
    while stack and nodes_explored < budget:
        u, path, path_visited = stack.pop()
        nodes_explored += 1
        
        for v in graph.neighbors(u):
            if v in forbidden:
                continue
            
            v_idx = vertex_to_idx[v]
            if path_visited[v_idx]:
                continue
            
            new_path = path + [v]
            
            if v == target:
                return new_path
            
            new_visited = path_visited[:]
            new_visited[v_idx] = True
            stack.append((v, new_path, new_visited))
    
    return None





# MAIN solver

def solve_some(graph):
    reach = reachable(graph, graph.s)
    if not reach.get(graph.t, False):
        return False, "unreachable"

    if graph.directed:
        if is_DAG(graph):
            ans = some_DAG(graph)
            return ans, "dag"
        else:
            return "?!", "cyclic" # NP-hard for directed-cyclic
    else:
        # ans = some_undirected(graph)
        # return ans, "max-flow"

        ans = some_undirected_maxflow(graph)
        return ans, "max-flow"

        # ans = some_general(graph, budget=1000)
        # return ans, "backtrack-undirected"

        #return '?!', 'undirected' # NP-hard for undirected
    


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

    print(f"\nResults written to {csv_file}")
