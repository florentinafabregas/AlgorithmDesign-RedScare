import sys 
import os
from parser import GraphParser

def bfs(graph, start, target, red_vertices):
    visited = set()
    queue = [(start, False)]
    visited.add(start)
    
    while queue:
        current, found_red = queue.pop(0)
        
        if current == target:
            if found_red:
                return True
        
        is_red = current in red_vertices
        
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, found_red or is_red))
    
    return False

data_folder = '../red-scare/data'
files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

for filename in files:
    filepath = os.path.join(data_folder, filename)
    g = GraphParser.from_file(filepath)
    
    result = bfs(g, g.s, g.t, g.red)
    
    if result:
        print(f'{filename}: true')
    else:
        print(f'{filename}: false')