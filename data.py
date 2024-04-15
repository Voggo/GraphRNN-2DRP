import numpy as np
import torch
import networkx as nx
from generator import generate_rects_and_graph


class DatasetSimple(torch.utils.data.Dataset):
    def __init__(self, num_graphs, height, width):
        self.n_breaks = 5
        self.data_adj = []
        self.data_bfs_adj = []
        self.max_num_nodes = 0

        self.num_graphs = num_graphs

        for _ in range(num_graphs):
            rects, adj, _, _ = generate_rects_and_graph(height, width, self.n_breaks)
            nodes = list(map(lambda x: x.get_as_node(), rects))
            self.max_num_nodes = max(self.max_num_nodes, len(nodes))
            self.data_adj.append(adj)
            bfs_index = self.bfs_index(adj, 0)
            self.data_bfs_adj.append(adj[np.ix_(bfs_index, bfs_index)])

        for i, bfs_adj in enumerate(self.data_bfs_adj):
            # Calculate padding size
            pad_size = self.max_num_nodes - bfs_adj.shape[0]

            # Pad the array
            if pad_size > 0:
                self.data_bfs_adj[i] = np.pad(
                    bfs_adj, ((0, pad_size), (0, pad_size)), mode="constant"
                )

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
        x = np.zeros((self.max_num_nodes + 1, self.max_num_nodes))
        y = np.zeros((self.max_num_nodes + 1, self.max_num_nodes))
        x[0, :] = np.ones((1, self.max_num_nodes))
        y[:-1, :] = self.data_bfs_adj[index]
        x[1:, :] = self.data_bfs_adj[index]
        y = torch.tensor(y)
        x = torch.tensor(x)
        return {"x": x, "y": y}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs, height, width):
        self.n_breaks = 5
        self.data_nodes = []
        self.data_nodes_width = []
        self.data_nodes_height = []
        self.data_bfs_nodes = []
        self.data_adj = []
        self.data_bfs_adj = []
        self.data_edge_dir = []
        self.data_bfs_edge_dir = []
        self.data_edge_angle = []
        self.data_bfs_edge_angle = []
        self.max_num_nodes = 0

        self.num_graphs = num_graphs

        for _ in range(num_graphs):
            rects, adj, edge_dir, edge_angle = generate_rects_and_graph(
                height, width, self.n_breaks
            )
            nodes = list(map(lambda x: x.get_as_node(), rects))
            self.max_num_nodes = max(self.max_num_nodes, len(nodes))

            self.data_nodes.append(nodes)
            self.data_adj.append(adj)
            self.data_edge_angle.append(edge_angle)
            self.data_edge_dir.append(edge_dir)

            bfs_index = self.bfs_index(adj, 0)
            self.data_bfs_nodes.append(np.array([nodes[i] for i in bfs_index]))
            self.data_bfs_adj.append(adj[np.ix_(bfs_index, bfs_index)])
            self.data_bfs_edge_dir.append(edge_dir[np.ix_(bfs_index, bfs_index)])
            self.data_bfs_edge_angle.append(edge_angle[np.ix_(bfs_index, bfs_index)])
        self.data_nodes_width = [0] * num_graphs
        self.data_nodes_height = [0] * num_graphs
        for i, bfs_adj in enumerate(self.data_bfs_adj):
            # Calculate padding size
            pad_size = self.max_num_nodes - bfs_adj.shape[0]
            self.data_nodes_width[i] = np.tile(
                self.data_bfs_nodes[i][:, 0], (len(self.data_bfs_nodes[i]), 1)
            ).T
            self.data_nodes_height[i] = np.tile(
                self.data_bfs_nodes[i][:, 1], (len(self.data_bfs_nodes[i]), 1)
            ).T
            # Pad the array
            if pad_size > 0:
                self.data_bfs_adj[i] = np.pad(
                    self.data_bfs_adj[i],
                    ((0, pad_size), (0, pad_size)),
                    mode="constant",
                )
                self.data_bfs_edge_dir[i] = np.pad(
                    self.data_bfs_edge_dir[i],
                    ((0, pad_size), (0, pad_size)),
                    mode="constant",
                )
                self.data_bfs_edge_angle[i] = np.pad(
                    self.data_bfs_edge_angle[i],
                    ((0, pad_size), (0, pad_size)),
                    mode="constant",
                )
                self.data_nodes_width[i] = np.pad(
                    self.data_nodes_width[i],
                    ((0, pad_size), (0, pad_size)),
                    mode="constant",
                )
                self.data_nodes_height[i] = np.pad(
                    self.data_nodes_height[i],
                    ((0, pad_size), (0, pad_size)),
                    mode="constant",
                )

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
        nodes = np.array(self.data_nodes[index])
        x = np.zeros((7, self.max_num_nodes + 1, self.max_num_nodes))
        y = np.zeros((3, self.max_num_nodes, self.max_num_nodes))
        x[:3, 0:1, :] = np.ones((3, 1, self.max_num_nodes))
        x[0, 1:, :] = np.tril(self.data_bfs_adj[index])
        y[0, :, :] = np.tril(self.data_bfs_adj[index])
        x[1, 1:, :] = np.tril(self.data_bfs_edge_dir[index])
        y[1, :, :] = np.tril(self.data_bfs_edge_dir[index])
        x[2, 1:, :] = np.tril(self.data_bfs_edge_angle[index])
        y[2, :, :] = np.tril(self.data_bfs_edge_angle[index])
        x[3, :-1, :] = np.tril(self.data_nodes_width[index])
        # x[3, 0, 1:] += 1 # add one to the first row but the first node in the graph
        # y[3, :, :] = np.tril(self.data_nodes_width[index])
        x[4, :-1, :] = np.tril(self.data_nodes_height[index])
        x[5, :-1, :] = np.tril(self.data_nodes_width[index].T)
        x[6, :-1, :] = np.tril(self.data_nodes_height[index].T)
        # x[4, 0, 1:] += 1 # add one to the first row but the first node in the graph
        # y[4, :, :] = np.tril(self.data_nodes_height[index])
        y = torch.tensor(y)
        x = torch.tensor(x)

        nodes = torch.tensor(nodes)
        return {"x": x, "y": y}


if __name__ == "__main__":
    data = Dataset(10, 100, 100)
    for i in range(10):
        d = data[i]
        x = d["x"]
        print(x)
        # print(x)
        print(d["y"].shape)
    # training_data = torch.utils.data.DataLoader(
    #     data, batch_size=2, shuffle=False
    # )
    # for batch in training_data:
    #     print(batch["x"].shape)
    #     print(batch["y"].shape)

