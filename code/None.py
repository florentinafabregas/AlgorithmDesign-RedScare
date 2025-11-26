from parser import GraphParser
import glob
from collections import deque
import csv
import os


def solveNone(g):
    if g.s == g.t:
        return 0

    if (g.t in g.neighbors(g.s)):
        return 1

    r_pruned = g.red - {g.s, g.t}

    distances = {g.s: 0}

    queue = deque([g.s])

    found = False

    while queue and not found:
        curr = queue.popleft()

        for next in g.neighbors(curr):
            if next in r_pruned:
                continue

            if next not in distances:
                distances[next] = distances[curr] + 1
                if next == g.t:
                    return distances[next]
                queue.append(next)
    
    return -1


if __name__ == "__main__":
    data_dir = "../red-scare/data"
    graph_files = sorted(glob.glob(f"{data_dir}/*.txt"))

    results = []
    # graph_file = "../red-scare/data/G-ex.txt"
    for filepath in graph_files:
        g = GraphParser.from_file(filepath)

        name = os.path.splitext(filepath.split("/")[-1])[0]

        try:
            ans = solveNone(g)
            results.append((name, g.n, ans))
            print(f"{name}: {ans}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"{name}: ERROR — {e}")

    # write CSV
    csv_file = "../results/none_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "n", "answer"])
        writer.writerows(results)
