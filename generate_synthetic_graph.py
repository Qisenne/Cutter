import os
import networkx as nx

num_nodes = 1000
m = 3

graph = nx.barabasi_albert_graph(n=num_nodes, m=m)

output_dir = "generated_graph"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "synthetic_graph.edgelist")

nx.write_edgelist(graph, output_path, data=False)
print(f"[INFO] Synthetic graph saved to: {output_path}")
