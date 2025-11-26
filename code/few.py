import sys
import os
import csv
import heapq
from parser import Graph, GraphParser


def solve_few(graph: Graph, s: str, t: str, red: set) -> int:
    if s == t:
        return 1 if s in red else 0

    dist = {v: float('inf') for v in graph.vertices}
    dist[s] = 1 if s in red else 0

    pq = [(dist[s], s)]

    while pq:
        cur_dist, u = heapq.heappop(pq)

        if cur_dist > dist[u]:
            continue
        if u == t:
            return cur_dist

        for v in graph.neighbors(u):
            cost = 1 if v in red else 0
            new_dist = cur_dist + cost

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return -1


def process_all_graphs(folder: str = '../red-scare/data', csv_name: str = 'few_results.csv'):
    results = []
    files = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))

    for filename in files:
        try:
            g = GraphParser.from_file(os.path.join(folder, filename))
            result = solve_few(g, g.s, g.t, g.red)
            results.append((filename, result))

            print(f"Completed {filename}")

        except Exception as e:
            print(f"ERROR processing {filename}: {e}")
            results.append((filename, "ERROR"))

    output_path = os.path.join("../results", csv_name)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "result"])
        writer.writerows(results)

    print(f"\nAll results saved to {csv_name}\n")
    return results


def main():
    if len(sys.argv) == 1:
        process_all_graphs()
        return

    target = sys.argv[1]
    if os.path.isdir(target):
        process_all_graphs(target)
    else:
        g = GraphParser.from_file(target)
        result = solve_few(g, g.s, g.t, g.red)
        print(result)


if __name__ == "__main__":
    main()

