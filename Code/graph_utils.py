import networkx as nx
import numpy as np
def pairwise_connectivity(graph):
    ans = 0
    components = nx.connected_components(graph)
    for component in components:
        n = len(component)
        ans += n * (n- 1) / 2
    return ans
def gen_graph(min_node_num, max_node_num):
    node_num = np.random.randint(max_node_num-min_node_num+1) + min_node_num
    return nx.barabasi_albert_graph(node_num, 3)
