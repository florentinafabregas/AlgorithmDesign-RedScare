#!/usr/bin/env python3

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

# Import your parser code (graph_io.py) here
from parser import GraphParser, Graph


def to_nx_graph(g: Graph) -> nx.Graph | nx.DiGraph:
    if g.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes with attributes
    for v in g.vertices:
        if v == g.s:
            color = 'green'
        elif v == g.t:
            color = 'blue'
        elif v in g.red:
            color = 'red'
        else:
            color = 'white'
        G.add_node(v, color=color)
    
    # Add edges
    for u, neighbors in g.adj.items():
        for v in neighbors:
            if g.directed or (u <= v):  # avoid double-adding undirected edges
                G.add_edge(u, v)
    
    return G


def draw_graph(g: Graph):
    G = to_nx_graph(g)
    node_colors = [data['color'] for _, data in G.nodes(data=True)]
    pos = nx.spring_layout(G, seed=42)  # consistent layout
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors='black', node_size=800)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>' if g.directed else '-', arrows=g.directed)
    nx.draw_networkx_labels(G, pos, font_color='black')
    plt.axis('off')
    plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph_file.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        g = GraphParser.from_file(str(path))
        print(f"Loaded graph: {g}")
        draw_graph(g)
    except Exception as e:
        print(f"Error loading or drawing graph: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
