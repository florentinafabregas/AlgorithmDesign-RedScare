import os
import sys
import re
from collections import defaultdict, deque
from parser import GraphParser
import time
import glob
from pathlib import Path
import pandas as pd

def build_alternating_subgraph(G):
    # normalize red set to strings
    red = {str(r) for r in G.red}
    
    # normalize vertices to strings as well (this works for word graphs too)
    vertices = [str(v) for v in G.vertices]

    # adjacency for the alternating subgraph
    sub_adj = {v: [] for v in vertices}

    for v in vertices:
        for u in G.neighbors(v):  # u is usually already a string
            u = str(u)
            # keep only edges where exactly one endpoint is red
            if (v in red) ^ (u in red):
                sub_adj[v].append(u)

    return sub_adj

def shortest_alternating_path(G):
    sub_adj = build_alternating_subgraph(G)
    
    s = str(G.s)
    t = str(G.t)
    
    dist = {s: 0}
    q = deque([s])
    
    while q:
        v = q.popleft()
        if v == t:
            return True, dist[v]
        
        for u in sub_adj.get(v, []):
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return False, None
        
        
data_dir = "../red-scare/data"
results = []

graph_files = sorted(glob.glob(f"{data_dir}/*.txt"))

for filepath in graph_files:
    instance_name = Path(filepath).stem     # correct cross-platform method
    G = GraphParser.from_file(filepath)
    ans = shortest_alternating_path(G)
    # collect row with instance_name
    results.append((instance_name, G.n, G.m, len(G.red), G.s, G.t, ans[0], ans[1]))

df_result = pd.DataFrame(results)
df_result.columns = ["instance", "n", "m", "r", "s", "t", "answer", "distance"]
df_result.to_csv("../results/alternate_results.csv")