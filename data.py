import numpy as np
import torch
import networkx as nx
from generator import generate_rects_and_graph


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs, height, width):
        self.n_breaks = 5
        self.data_nodes = []
        self.data_bfs_nodes = []
        self.data_adj = []
        self.data_bfs_adj = []
        self.data_edge_dir = []
        self.data_edge_angle = []

        self.num_graphs = num_graphs

        for _ in range(num_graphs):
            rects, adj, edge_dir, edge_angle = generate_rects_and_graph(
                height, width, self.n_breaks
            )
            nodes = list(map(lambda x: x.get_as_node(), rects))
            self.data_nodes.append(nodes)
            self.data_adj.append(adj)
            self.data_edge_angle.append(edge_angle)
            self.data_edge_dir.append(edge_dir)

            bfs_index = self.bfs_index(adj, 0)
            self.data_bfs_nodes.append([nodes[i] for i in bfs_index])
            self.data_bfs_adj.append(adj[np.ix_(bfs_index, bfs_index)])

    def bfs_index(self, adj, start):
        """return the index of the nodes in bfs order"""
        graph = nx.from_numpy_array(adj)
        bfs_edges = nx.bfs_edges(graph, start)
        nodes = [start] + [v for u, v in bfs_edges]
        print(nodes)
        return nodes

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, index):
        x = torch.tensor(self.data_nodes[index])
        y = {}
        y["adj"] = torch.tensor(self.data_adj[index])
        # y["edge_dir"] = torch.tensor(self.data_edge_dir[index])
        y["edge_angle"] = torch.tensor(self.data_edge_angle[index])
        return x, y


if __name__ == "__main__":
    data = Dataset(10, 100, 100)
    x, y = data[0]
    adj = y["adj"]