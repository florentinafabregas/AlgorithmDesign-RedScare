from parser import GraphParser
import glob
from collections import deque
import csv
import os


def solveNone(g):
    """
    Implementation of the None problem solution through BFS
    
    :param g: graph instance object
    """
    if g.s == g.t:
        return 0

    # trivial case where s and t are directly connected
    if (g.t in g.neighbors(g.s)):
        return 1

    # nodes s and t do not fall under the red-avoidance constraint
    r_pruned = g.red - {g.s, g.t}

    distances = {g.s: 0}

    queue = deque([g.s])

    found = False

    while queue and not found:
        curr = queue.popleft()

        for next in g.neighbors(curr):
            # we avoid any internal red nodes
            if next in r_pruned:
                continue

            if next not in distances:
                distances[next] = distances[curr] + 1
                # if node t is reached, BSF found the shortest valid path
                if next == g.t:
                    return distances[next]
                queue.append(next)
    
    # no such path exists
    return -1


if __name__ == "__main__":
    data_dir = "../red-scare/data"
    graph_files = sorted(glob.glob(f"{data_dir}/*.txt"))

    results = []
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

    # write CSV results
    csv_file = "../results/none_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "n", "answer"])
        writer.writerows(results)
