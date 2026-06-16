

class Node:
    def __init__(self, id, x, weight, bias, connected_nodes: list):
        self.id = id
        self.x = x
        self.weight = weight
        self.bias = bias
        self.connected_nodes = connected_nodes

    def get_id(self):
        return self.id

    def get_data(self):
        return self.x
    
    def get_weight(self):
        return self.weight
    
    def get_bias(self):
        return self.bias

    def get_connected_nodes(self):
        return self.connected_nodes
    

