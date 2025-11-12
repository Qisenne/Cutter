import os
import networkx as nx

# === 生成图参数设置 ===
num_nodes = 1000  # 节点数
m = 3             # 每个新节点连接m个已有节点

# === 生成BA图（可替换为ER图等）===
graph = nx.barabasi_albert_graph(n=num_nodes, m=m)

# === 保存路径设置 ===
output_dir = "generated_graph"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "synthetic_graph.edgelist")

# === 保存图结构 ===
nx.write_edgelist(graph, output_path, data=False)
print(f"[INFO] Synthetic graph saved to: {output_path}")
