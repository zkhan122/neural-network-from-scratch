from typing import List
from node import Node

class Layer():
    def __init__(self, layer_i, nodes: List[Node]):
        self.layer_i = layer_i
        self.nodes = nodes
    
    def get_nodes_in_layer(self):
        node_verbose_buffer = []
        for node in self.nodes:
            node_verbose_buffer.append(tuple(node.get_id, node_verbose_buffer))
        return node_verbose_buffer

    def print_nodes_in_layer(self):
        print(f"Layer {self.layer_i} with node count {len(self.get_nodes_in_layer)} -> Nodes: {self.get_nodes_in_layer}")


