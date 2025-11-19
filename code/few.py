from collections import deque
from parser import Graph, GraphParser
import sys
import os


def solve_few(graph: Graph, s: str, t: str, red_vertices: set) -> int:
    """Return the minimum number of red vertices on any path from s to t."""
    if s == t:
        return 1 if s in red_vertices else 0

    dist = {v: float('inf') for v in graph.vertices}
    dist[s] = 1 if s in red_vertices else 0
    dq = deque([s])

    while dq:
        u = dq.popleft()
        if u == t:
            return dist[t]

        for v in graph.neighbors(u):
            cost = 1 if v in red_vertices else 0
            new_dist = dist[u] + cost

            if new_dist < dist[v]:
                dist[v] = new_dist
                (dq.appendleft if cost == 0 else dq.append)(v)

    return -1


def process_all_graphs(data_folder: str = '../red-scare/data'):
    files = sorted(f for f in os.listdir(data_folder) if f.endswith('.txt'))
    results = []

    for filename in files:
        path = os.path.join(data_folder, filename)
        try:
            g = GraphParser.from_file(path)
            result = solve_few(g, g.s, g.t, g.red)
            print(f"{filename}: {result}")
            results.append((filename, result))
        except Exception as e:
            print(f"{filename}: ERROR - {e}")

    return results


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]

        if os.path.isdir(target):
            process_all_graphs(target)
        else:
            g = GraphParser.from_file(target)
            result = solve_few(g, g.s, g.t, g.red)
            print(result)
    else:
        process_all_graphs()


if __name__ == "__main__":
    main()

