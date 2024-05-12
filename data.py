import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from generator import generate_rects_and_graph, convert_graph_to_rects


def get_bfs_index(adj, start):
    """return the index of the nodes in bfs order"""
    graph = nx.from_numpy_array(adj)
    bfs_edges = nx.bfs_edges(graph, start)
    nodes = [start] + [v for u, v in bfs_edges]
    print(nodes)
    return nodes


def generate_datasets(num_graphs, height, width, test=False, n_breaks=5):
    max_num_nodes = 12
    data_bfs_nodes = {i: [] for i in range(4, max_num_nodes + 1)}
    data_nodes_width = {i: [] for i in range(4, max_num_nodes + 1)}
    data_nodes_height = {i: [] for i in range(4, max_num_nodes + 1)}
    data_bfs_adj = {i: [] for i in range(4, max_num_nodes + 1)}
    data_bfs_edge_dir = {i: [] for i in range(4, max_num_nodes + 1)}
    data_bfs_offset = {i: [] for i in range(4, max_num_nodes + 1)}
    is_full = {i: False for i in range(4, max_num_nodes + 1)}
    while True:
        rects, adj, edge_dir, offset = generate_rects_and_graph(
            height, width, n_breaks
        )
        nodes_len = len(rects)
        if nodes_len < 4 or nodes_len > max_num_nodes or is_full[nodes_len]:
            continue
        nodes = list(map(lambda x: x.get_as_node(), rects))
        bfs_index = get_bfs_index(adj, 0)
        bfs_nodes = [nodes[i] for i in bfs_index]
        data_bfs_nodes[nodes_len].append(np.array(bfs_nodes))
        data_nodes_width[nodes_len].append(
            np.tile(
                data_bfs_nodes[nodes_len][-1][:, 0],
                (len(data_bfs_nodes[nodes_len][-1]), 1),
            ).T
        )
        data_nodes_height[nodes_len].append(
            np.tile(
                data_bfs_nodes[nodes_len][-1][:, 1],
                (len(data_bfs_nodes[nodes_len][-1]), 1),
            ).T
        )
        data_bfs_adj[nodes_len].append(adj[np.ix_(bfs_index, bfs_index)])
        data_bfs_edge_dir[nodes_len].append(edge_dir[np.ix_(bfs_index, bfs_index)])
        data_bfs_offset[nodes_len].append(offset[np.ix_(bfs_index, bfs_index)])

        if len(data_bfs_nodes[nodes_len]) == num_graphs:
            is_full[nodes_len] = True
            if all(is_full.values()):
                break
    print("Done generating datasets")
    for i in range(4, max_num_nodes + 1):
        data_bfs_nodes[i] = np.array(data_bfs_nodes[i])
        data_nodes_width[i] = np.array(data_nodes_width[i])
        data_nodes_height[i] = np.array(data_nodes_height[i])
        data_bfs_adj[i] = np.array(data_bfs_adj[i])
        data_bfs_edge_dir[i] = np.array(data_bfs_edge_dir[i])
        data_bfs_offset[i] = np.array(data_bfs_offset[i])
        suffix = "_test" if test else ""
        np.save(f"datasets/data_bfs_nodes_{i}{suffix}.npy", data_bfs_nodes[i])
        np.save(f"datasets/data_nodes_width_{i}{suffix}.npy", data_nodes_width[i])
        np.save(f"datasets/data_nodes_height_{i}{suffix}.npy", data_nodes_height[i])
        np.save(f"datasets/data_bfs_adj_{i}{suffix}.npy", data_bfs_adj[i])
        np.save(f"datasets/data_bfs_edge_dir_{i}{suffix}.npy", data_bfs_edge_dir[i])
        np.save(f"datasets/data_bfs_offset_{i}{suffix}.npy", data_bfs_offset[i])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, node_len, test=False):
        suffix = "_test" if test else ""
        self.data_bfs_nodes = np.load(f"datasets/data_bfs_nodes_{node_len}{suffix}.npy")
        self.data_nodes_width = np.load(
            f"datasets/data_nodes_width_{node_len}{suffix}.npy"
        )
        self.data_nodes_height = np.load(
            f"datasets/data_nodes_height_{node_len}{suffix}.npy"
        )
        self.data_bfs_adj = np.load(f"datasets/data_bfs_adj_{node_len}{suffix}.npy")
        self.data_bfs_edge_dir = torch.LongTensor(
            np.load(f"datasets/data_bfs_edge_dir_{node_len}{suffix}.npy")
        )
        self.data_bfs_offset = np.load(
            f"datasets/data_bfs_offset_{node_len}{suffix}.npy"
        )
        self.max_num_nodes = node_len
        self.num_graphs = len(self.data_bfs_nodes)

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
        x = np.zeros((11, self.max_num_nodes, self.max_num_nodes))
        y = np.zeros((7, self.max_num_nodes - 1, self.max_num_nodes))
        edge_dir = np.array(F.one_hot(self.data_bfs_edge_dir[index], 5))
        x[:, 0:1, :] = np.ones((11, 1, self.max_num_nodes))
        x[0, 1:, :] = np.tril(self.data_bfs_adj[index][1:, :])
        y[0, :, :] = np.tril(self.data_bfs_adj[index][1:, :])
        x[1, 1:, :] = np.tril(edge_dir[1:, :, 0])
        y[1, :, :] = np.tril(edge_dir[1:, :, 0])
        x[2, 1:, :] = np.tril(edge_dir[1:, :, 1])
        y[2, :, :] = np.tril(edge_dir[1:, :, 1])
        x[3, 1:, :] = np.tril(edge_dir[1:, :, 2])
        y[3, :, :] = np.tril(edge_dir[1:, :, 2])
        x[4, 1:, :] = np.tril(edge_dir[1:, :, 3])
        y[4, :, :] = np.tril(edge_dir[1:, :, 3])
        x[5, 1:, :] = np.tril(edge_dir[1:, :, 4])
        y[5, :, :] = np.tril(edge_dir[1:, :, 4])

        x[6, 1:, :] = np.tril(self.data_bfs_offset[index][1:, :])
        y[6, :, :] = np.tril(self.data_bfs_offset[index][1:, :])
        x[7, :, :] = np.tril(self.data_nodes_width[index].T)
        x[8, :, :] = np.tril(self.data_nodes_height[index].T)
        x[9, :, :] = np.tril(self.data_nodes_width[index])
        x[10, :, :] = np.tril(self.data_nodes_height[index])
        y = torch.tensor(y)
        x = torch.tensor(x)
        return {"x": x, "y": y}


if __name__ == "__main__":
    generate_datasets(250, 100, 100, test=False)
    data = Dataset(6, test=False)
    x = data[0]["x"]
    print(x)
    # for i in range(10):
    #     d = data[i]
    #     x = d["x"]
    #     print(x)
    #     # print(x)
    #     print(d["y"].shape)
    # training_data = torch.utils.data.DataLoader(
    #     data, batch_size=2, shuffle=False
    # )
    # for batch in training_data:
    #     print(batch["x"].shape)
    #     print(batch["y"].shape)
