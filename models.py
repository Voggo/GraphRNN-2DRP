import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
import numpy as np

from data import Dataset
from plot_rects import plot_rects
from generator import convert_graph_to_rects
from dataclasses_rect_point import Rectangle, Point
from utils import convert_center_to_lower_left, sample_graph
from eval_solutions import evaluate_solution

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        num_layers,
        output_size=1,
        output_hidden_size=16,
    ):
        super(RNN, self).__init__()

        self.num_layers = num_layers

        self.embedding = torch.nn.Linear(input_size, embedding_size)
        self.relu = torch.nn.ReLU()
        self.rnn = torch.nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, output_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_hidden_size, output_size),
        )
        self.hidden = None

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size)

    def forward(self, x):
        input = self.embedding(x)
        input = self.relu(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output_embed = self.output(output.clone())
        return output, output_embed


def train_rnn(
    rnn_graph,
    rnn_edge,
    device,
    learning_rate,
    learning_rate_steps,
    lambda_ratios,
    epochs,
    num_nodes,
    training_data,
    model_dir_name,
):
    optimizer_rnn_graph = torch.optim.Adam(
        list(rnn_graph.parameters()), lr=learning_rate
    )
    optimizer_rnn_edge = torch.optim.Adam(list(rnn_edge.parameters()), lr=learning_rate)
    scheduler_rnn_graph = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn_graph, learning_rate_steps, gamma=0.2
    )
    scheduler_rnn_edge = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn_edge, learning_rate_steps, gamma=0.2
    )
    rnn_graph.train()
    rnn_edge.train()
    losses = []
    losses_bce_adj = []
    losses_bce_dir = []
    losses_mae = []
    for epoch in range(epochs):
        loss_sum = 0
        loss_sum_adj = 0
        loss_sum_dir = 0
        loss_sum_l1 = 0
        batch_size = 0
        for batch in training_data:
            batch_size = batch["x"].size(0)
            rnn_graph.zero_grad()
            rnn_edge.zero_grad()
            x_bumpy = batch["x"].float().to(device)
            y_bumpy = batch["y"].float().to(device)
            x = torch.cat(
                (
                    x_bumpy[:, 0, :, :],
                    x_bumpy[:, 1, :, :],
                    x_bumpy[:, 2, :, :],
                    x_bumpy[:, 3, :, :],
                    x_bumpy[:, 4, :, :],
                    x_bumpy[:, 5, :, :],
                    x_bumpy[:, 6, :, :],
                    x_bumpy[:, 7, :, :],
                    x_bumpy[:, 8, :, :],
                ),
                dim=2,
            ).to(device)
            y = torch.cat(
                (
                    y_bumpy[:, 0, :, :],
                    y_bumpy[:, 1, :, :],
                    y_bumpy[:, 2, :, :],
                    y_bumpy[:, 3, :, :],
                    y_bumpy[:, 4, :, :],
                    y_bumpy[:, 5, :, :],
                    y_bumpy[:, 6, :, :],
                ),
                dim=2,
            ).to(device)
            rnn_graph.hidden = rnn_graph.init_hidden(x.size(0)).to(device)
            _, output_embed = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), num_nodes - 1, num_nodes * 7).to(device)
            for i in range(num_nodes - 1):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0)).to(device)
                rnn_edge.hidden[0, :, :] = output_embed[:, i, :].to(device)
                edge_input_graph = (
                    torch.stack([x_bumpy[:, j, i + 1, :-1] for j in range(7)], dim=1)
                    .transpose(-2, -1)
                    .to(device)
                )
                edge_input_nodes = (
                    torch.stack([x_bumpy[:, j, i, :-1] for j in range(7, 11)], dim=1)
                    .transpose(-2, -1)
                    .to(device)
                )
                edge_input_graph = torch.cat(
                    (torch.ones((batch_size, 1, 7)).to(device), edge_input_graph), dim=1
                )
                edge_input_nodes = torch.cat(
                    (edge_input_nodes, torch.zeros((batch_size, 1, 4)).to(device)),
                    dim=1,
                )
                edge_input = torch.cat((edge_input_graph, edge_input_nodes), dim=2).to(
                    device
                )
                # (batch_size, seq_len(node_len), features)
                _, output_edge = rnn_edge(edge_input.clone())
                output_edge = output_edge[:, :, :]
                output_edge[:, :, 0:6] = F.sigmoid(output_edge[:, :, 0:6])
                output_edge = torch.cat(
                    (
                        output_edge[:, :, 0:1].mT,
                        output_edge[:, :, 1:2].mT,
                        output_edge[:, :, 2:3].mT,
                        output_edge[:, :, 3:4].mT,
                        output_edge[:, :, 4:5].mT,
                        output_edge[:, :, 5:6].mT,
                        output_edge[:, :, 6:].mT,
                    ),
                    dim=2,
                )
                y_pred[:, i, :] = output_edge[:, 0, :]
            y_pred_adj = torch.tril(y_pred[:, :, :num_nodes].clone())
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            y_pred[:, :, num_nodes : num_nodes * 2] *= torch.tril(1 - y_pred_adj)
            y_pred[:, :, :num_nodes] = y_pred_adj
            loss_kl_adj = F.binary_cross_entropy(
                y_pred[:, :, :num_nodes], y[:, :, :num_nodes]
            )
            loss_kl_dir = F.binary_cross_entropy(
                y_pred[:, :, num_nodes : num_nodes * 6],
                y[:, :, num_nodes : num_nodes * 6],
            )
            loss_l1 = F.l1_loss(y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 6 :])
            loss = (
                lambda_ratios["kl_adj"] * loss_kl_adj
                + lambda_ratios["kl_dir"] * loss_kl_dir
                + lambda_ratios["l1"] * loss_l1
            )
            loss_sum += loss.item()
            loss_sum_adj += loss_kl_adj.item()
            loss_sum_dir += loss_kl_dir.item()
            loss_sum_l1 += loss_l1.item()
            loss.backward()
            optimizer_rnn_graph.step()
            optimizer_rnn_edge.step()
        scheduler_rnn_graph.step()
        scheduler_rnn_edge.step()
        batches = len(training_data) / batch_size
        losses.append(loss_sum / batches)
        losses_bce_adj.append(loss_sum_adj / batches)
        losses_bce_dir.append(loss_sum_dir / batches)
        losses_mae.append(loss_sum_l1 / batches)
        if epoch % 5 == 0:
            print(
                f"epoch: {epoch}, loss: {losses[-1]}, bce adj: {losses_bce_adj[-1]}, bce dir: {losses_bce_dir[-1]}, mae: {losses_mae[-1]}"
            )
        if epoch % 100 == 0 and not epoch == 0:
            torch.save(
                rnn_graph.state_dict(),
                f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
            )
            torch.save(
                rnn_edge.state_dict(),
                f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
            )

    torch.save(
        rnn_graph.state_dict(),
        f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
    )
    torch.save(
        rnn_edge.state_dict(),
        f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
    )
    return rnn_graph, rnn_edge, losses, losses_bce_adj, losses_bce_dir, losses_mae


def test_rnn(device, num_nodes, model_dir_name, test_data):
    hp = {}
    with open(
        f"models/{model_dir_name}_graph_size_{num_nodes}/hyperparameters.json", "r"
    ) as f:
        hp = json.load(f)
    rnn_graph = RNN(
        9 * num_nodes,
        4 * num_nodes,
        hp["hidden_size_1"],
        hp["num_layers"],
        output_size=hp["hidden_size_2"],
    ).to(device)
    rnn_edge = RNN(
        11,
        16,
        hp["hidden_size_2"],
        hp["num_layers"],
        output_size=7,
    )
    rnn_graph.load_state_dict(
        torch.load(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
            map_location=torch.device("cpu"),
        )
    )
    rnn_edge.load_state_dict(
        torch.load(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
            map_location=torch.device("cpu"),
        )
    )
    rnn_graph.eval()
    rnn_edge.eval()
    with torch.no_grad():
        # losses pr batch in the dataset
        losses = []
        losses_bce_adj = []
        losses_bce_dir = []
        losses_mae = []
        for batch in test_data:
            x_bumpy = batch["x"].float().to(device)
            y_bumpy = batch["y"].float().to(device)
            x = torch.cat(
                (
                    x_bumpy[:, 0, :, :],
                    x_bumpy[:, 1, :, :],
                    x_bumpy[:, 2, :, :],
                    x_bumpy[:, 3, :, :],
                    x_bumpy[:, 4, :, :],
                    x_bumpy[:, 5, :, :],
                    x_bumpy[:, 6, :, :],
                    x_bumpy[:, 7, :, :],
                    x_bumpy[:, 8, :, :],
                ),
                dim=2,
            ).to(device)
            y = torch.cat(
                (
                    y_bumpy[:, 0, :, :],
                    y_bumpy[:, 1, :, :],
                    y_bumpy[:, 2, :, :],
                    y_bumpy[:, 3, :, :],
                    y_bumpy[:, 4, :, :],
                    y_bumpy[:, 5, :, :],
                    y_bumpy[:, 6, :, :],
                ),
                dim=2,
            ).to(device)
            rnn_graph.hidden = rnn_graph.init_hidden(x.size(0)).to(device)
            _, output_embed = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), num_nodes - 1, num_nodes * 7).to(device)
            for i in range(num_nodes - 1):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0)).to(device)
                rnn_edge.hidden[0, :, :] = output_embed[:, i, :].to(device)
                edge_input_graph = (
                    torch.stack([x_bumpy[:, j, i + 1, :-1] for j in range(7)], dim=1)
                    .transpose(-2, -1)
                    .to(device)
                )
                edge_input_nodes = (
                    torch.stack([x_bumpy[:, j, i, :-1] for j in range(7, 11)], dim=1)
                    .transpose(-2, -1)
                    .to(device)
                )
                edge_input_graph = torch.cat(
                    (torch.ones((batch_size, 1, 7)).to(device), edge_input_graph), dim=1
                )
                edge_input_nodes = torch.cat(
                    (edge_input_nodes, torch.zeros((batch_size, 1, 4)).to(device)),
                    dim=1,
                )
                edge_input = torch.cat((edge_input_graph, edge_input_nodes), dim=2).to(
                    device
                )
                # (batch_size, seq_len(node_len), features)
                _, output_edge = rnn_edge(edge_input.clone())
                output_edge = output_edge[:, :, :]
                output_edge[:, :, 0:6] = F.sigmoid(output_edge[:, :, 0:6])
                output_edge = torch.cat(
                    (
                        output_edge[:, :, 0:1].mT,
                        output_edge[:, :, 1:2].mT,
                        output_edge[:, :, 2:3].mT,
                        output_edge[:, :, 3:4].mT,
                        output_edge[:, :, 4:5].mT,
                        output_edge[:, :, 5:6].mT,
                        output_edge[:, :, 6:].mT,
                    ),
                    dim=2,
                )
                y_pred[:, i, :] = output_edge[:, 0, :]
            y_pred_adj = torch.tril(y_pred[:, :, :num_nodes].clone())
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            y_pred[:, :, num_nodes : num_nodes * 2] *= torch.tril(1 - y_pred_adj)
            y_pred[:, :, :num_nodes] = y_pred_adj

            loss_kl_adj = F.binary_cross_entropy(
                y_pred[:, :, :num_nodes], y[:, :, :num_nodes]
            )
            loss_kl_dir = F.binary_cross_entropy(
                y_pred[:, :, num_nodes : num_nodes * 6],
                y[:, :, num_nodes : num_nodes * 6],
            )
            loss_l1 = F.l1_loss(y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 6 :])
            loss = loss_kl_adj + loss_kl_dir + loss_l1
            losses.append(loss.item())
            losses_bce_adj.append(loss_kl_adj.item())
            losses_bce_dir.append(loss_kl_dir.item())
            losses_mae.append(loss_l1.item())
            print(
                f"loss: {loss.item()}, bce adj: {loss_kl_adj.item()}, bce dir: {loss_kl_dir.item()}, mae: {loss_l1.item()}"
            )
            y_pred = torch.round(y_pred)
            # number of cells predicted is only lower triangle
            cells = 0.5 * (num_nodes - 1) * num_nodes
            accuracy_adj = (
                torch.sum(y_pred[:, :, :num_nodes] == y[:, :, :num_nodes]) - cells
            ) / cells
            accuracy_dir_0 = (
                torch.sum(
                    y_pred[:, :, num_nodes : num_nodes * 2]
                    == y[:, :, num_nodes : num_nodes * 2]
                )
                - cells
            ) / cells
            accuracy_dir_1 = (
                torch.sum(
                    y_pred[:, :, num_nodes * 2 : num_nodes * 3]
                    == y[:, :, num_nodes * 2 : num_nodes * 3]
                )
                - cells
            ) / cells
            accuracy_dir_2 = (
                torch.sum(
                    y_pred[:, :, num_nodes * 3 : num_nodes * 4]
                    == y[:, :, num_nodes * 3 : num_nodes * 4]
                )
                - cells
            ) / cells
            accuracy_dir_3 = (
                torch.sum(
                    y_pred[:, :, num_nodes * 4 : num_nodes * 5]
                    == y[:, :, num_nodes * 4 : num_nodes * 5]
                )
                - cells
            ) / cells
            accuracy_dir_4 = (
                torch.sum(
                    y_pred[:, :, num_nodes * 5 : num_nodes * 6]
                    == y[:, :, num_nodes * 5 : num_nodes * 6]
                )
                - cells
            ) / cells
            accuracy_dir = (
                accuracy_dir_0
                + accuracy_dir_1
                + accuracy_dir_2
                + accuracy_dir_3
                + accuracy_dir_4
            ) / 5
            print(
                f"accuracy adj: {accuracy_adj}, accuracy dir_0: {accuracy_dir_0}, accuracy dir_1: {accuracy_dir_1}, accuracy dir_2: {accuracy_dir_2}, accuracy dir_3: {accuracy_dir_3}, accuracy dir_4: {accuracy_dir_4}, accuracy dir: {accuracy_dir}"
            )
        losses_mean = sum(losses) / len(losses)
        losses_bce_adj_mean = sum(losses_bce_adj) / len(losses_bce_adj)
        losses_bce_dir_mean = sum(losses_bce_dir) / len(losses_bce_dir)
        losses_mse_mean = sum(losses_mae) / len(losses_mae)
        print(
            f"mean loss: {losses_mean}",
            f"mean bce adj: {losses_bce_adj_mean}, mean bce dir: {losses_bce_dir_mean}, mean mae: {losses_mse_mean}",
        )
        losses = {
            "losses": losses,
            "losses_bce_adj": losses_bce_adj,
            "losses_bce_dir": losses_bce_dir,
            "losses_mae": losses_mae,
        }
        json.dump(
            losses,
            open(
                f"models/{model_dir_name}_graph_size_{num_nodes}/test_losses.json", "w"
            ),
        )


def test_inference_rnn(
    device,
    hidden_size_1,
    hidden_size_2,
    num_nodes,
    graph,
    num_layers,
    model_dir_name,
    data,
    batch_size=1,
):
    rnn_graph = RNN(
        num_nodes * 9,
        num_nodes * 4,
        hidden_size_1,
        num_layers,
        output_size=hidden_size_2,
    ).to(device)
    rnn_edge = RNN(11, 16, hidden_size_2, num_layers, output_size=7).to(device)
    rnn_graph.load_state_dict(
        torch.load(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
            map_location=torch.device("cpu"),
        )
    )
    rnn_edge.load_state_dict(
        torch.load(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
            map_location=torch.device("cpu"),
        )
    )
    with torch.no_grad():
        rnn_graph.eval()
        rnn_edge.eval()
        print(data.data_bfs_adj[graph])
        print(data.data_bfs_edge_dir[graph])
        print(data.data_bfs_offset[graph])
        nodes = torch.tensor(data.data_bfs_nodes[graph]).to(device)
        x_step = torch.ones(batch_size, 1, num_nodes * 9).to(device)
        x_step[:, 0, 7 * num_nodes :] = torch.zeros(batch_size, 1, 2 * num_nodes).to(
            device
        )
        x_step[:, 0, 7 * num_nodes] = nodes[0][0]
        x_step[:, 0, 8 * num_nodes] = nodes[0][1]
        x_step_all = torch.zeros(batch_size, num_nodes, num_nodes * 9).to(
            device
        )  # debugging
        x_step_all[:, 0, :] = x_step
        y_pred = torch.zeros(batch_size, num_nodes - 1, num_nodes * 7).to(device)
        for i in range(num_nodes - 1):
            _, output_embed = rnn_graph(x_step)
            rnn_edge.hidden = rnn_edge.init_hidden(batch_size).to(device)
            rnn_edge.hidden[0, :, :] = output_embed[:, -1, :]
            edge_input_step = torch.ones(batch_size, 1, 11).to(device)
            edge_input_step_all = torch.zeros(batch_size, num_nodes, 11).to(device)
            edge_input_step[:, 0, 7] = nodes[0][0]
            edge_input_step[:, 0, 8] = nodes[0][1]
            edge_input_step[:, 0, 9] = nodes[i][0]
            edge_input_step[:, 0, 10] = nodes[i][1]
            edge_y_pred = torch.zeros(batch_size, num_nodes, 9).to(device)
            for j in range(i + 1):
                edge_input_step_all[:, j, :] = edge_input_step
                _, output_edge = rnn_edge(edge_input_step)
                output_edge[:, 0, 0:1] = torch.bernoulli(
                    F.sigmoid(output_edge[:, 0, 0:1])
                )
                dir = 0
                if output_edge[:, 0, 0] == 0:
                    dir = torch.argmax(F.sigmoid(output_edge[:, 0, 1:6]), 1)
                else:
                    dir = torch.argmax(F.sigmoid(output_edge[:, 0, 2:6]), 1)
                output_edge[:, 0, 1 + dir] = 1
                output_edge[:, 0, 1 : dir + 1] = 0
                output_edge[:, 0, dir + 2 : 6] = 0
                output_edge[:, 0, 0:] *= output_edge[:, 0, 0:1].clone()
                edge_input_step[:, :, :7] = output_edge[0, 0, :]
                if j < len(nodes) - 1:
                    edge_input_step[:, 0, 7] = nodes[j + 1][0]
                    edge_input_step[:, 0, 8] = nodes[j + 1][1]
                else:
                    edge_input_step[:, 0, 7] = 0
                    edge_input_step[:, 0, 8] = 0
                if i < len(nodes) - 1:
                    edge_input_step[:, 0, 9] = nodes[i][0]
                    edge_input_step[:, 0, 10] = nodes[i][1]
                else:
                    edge_input_step[:, 0, 9] = 0
                    edge_input_step[:, 0, 10] = 0
                edge_y_pred[:, j, :7] = output_edge[:, :, :7]
                edge_y_pred[:, j + 1, 7:] = edge_input_step[:, :, 7:9]
            edge_y_pred = edge_y_pred.transpose(-2, -1).flatten(start_dim=1, end_dim=2)
            y_pred[:, i, :] = edge_y_pred[:, : num_nodes * 7]
            x_step = edge_y_pred.unsqueeze(0).to(device)
            x_step[:, :, 7 * num_nodes] = nodes[0][0]
            x_step[:, :, 8 * num_nodes] = nodes[0][1]
            x_step_all[:, i + 1, :] = x_step
        y_pred = torch.cat((torch.zeros((1, 1, num_nodes * 7)), y_pred), dim=1)
        adj = y_pred[0, :, :num_nodes].reshape(num_nodes, num_nodes).to(torch.int64)
        adj.diagonal().fill_(0)
        adj = adj + adj.T.to(torch.int64)
        edge_dir = y_pred[0, :, num_nodes : num_nodes * 6]
        edge_dir_splits = torch.split(edge_dir, edge_dir.size(1) // 5, dim=1)
        edge_dir = torch.stack(edge_dir_splits, dim=2)
        edge_dir = torch.argmax(edge_dir, dim=2)
        edge_dir.diagonal().fill_(0)
        mapping = torch.tensor([0, 3, 4, 1, 2])
        edge_dir = (edge_dir + mapping[edge_dir].T).to(torch.int64)
        offset = y_pred[0, :, num_nodes * 6 :].reshape(num_nodes, num_nodes)
        offset.diagonal().fill_(0)
        offset = offset + offset.T
        print("current adj:")
        print(adj)
        print("current edge dir:")
        print(edge_dir)
        print("current offset:")
        print(offset)
        # Convert edge_dir to boolean tensor where True indicates the presence of an edge
        edge_mask = edge_dir > 0
        # Use the mask to filter the adjacency matrix
        adj = torch.where(edge_mask, adj, torch.zeros_like(adj))
        best_rects = None
        max_fill_ratio = 0
        min_overlap_area = 100000000
        for _ in range(1000):
            nodes_rects = [
                Rectangle(node[0].item(), node[1].item(), 0) for node in nodes
            ]
            sampled_graph = sample_graph(adj.numpy())
            # sampled_graph = adj.numpy()
            rects = convert_graph_to_rects(
                nodes_rects, sampled_graph, edge_dir.numpy(), offset.numpy()
            )
            rects = convert_center_to_lower_left(rects)
            fill_ratio, overlap_area = evaluate_solution(rects, 100, 100)
            if fill_ratio > max_fill_ratio:
                max_fill_ratio = fill_ratio
                best_rects = rects
                min_overlap_area = overlap_area
                # best_rects = rects
        print(f"max fill ratio: {max_fill_ratio}")
        print(f"min overlap area: {min_overlap_area}")
        plot_rects(
            best_rects,
            ax_lim=100,
            ay_lim=100,
            ax_min=-50,
            ay_min=-50,
            filename="rnn_rnn.png",
        )


def plot_losses(
    losses, losses_bce_adj, losses_bce_dir, losses_mse, model_dir_name, num_nodes
):
    # Create a subplot with 2 columns and 2 rows
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Losses")
    axs[0, 0].plot(losses, color="red")
    axs[0, 0].set_title("Total Loss", fontsize=10)
    axs[0, 1].plot(losses_bce_adj, color="green")
    axs[0, 1].set_title("BCE Adj Loss", fontsize=10)
    axs[1, 0].plot(losses_bce_dir, color="blue")
    axs[1, 0].set_title("BCE Dir Loss", fontsize=10)
    axs[1, 1].plot(losses_mse, color="orange")
    axs[1, 1].set_title("MSE Loss", fontsize=10)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(f"models/{model_dir_name}_graph_size_{num_nodes}/losses.png")
    plt.show()


def train(
    rnn_graph,
    rnn_edge,
    device,
    learning_rate,
    learning_rate_steps,
    lambda_ratios,
    epochs,
    num_nodes,
    training_data,
    model_dir_name,
):
    print(f"max nodes: {num_nodes}")
    override = False
    if (
        os.path.isfile(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth"
        )
        or os.path.isfile(
            f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth"
        )
        or override
    ):
        input("Models already exist. Press enter to continue")
    rnn_graph, rnn_edge, losses, losses_bce_adj, losses_bce_dir, losses_mse = train_rnn(
        rnn_graph,
        rnn_edge,
        device,
        learning_rate,
        learning_rate_steps,
        lambda_ratios,
        epochs,
        num_nodes,
        training_data,
        model_dir_name,
    )
    plot_losses(
        losses, losses_bce_adj, losses_bce_dir, losses_mse, model_dir_name, num_nodes
    )
    losses = {
        "losses": losses,
        "losses_bce_adj": losses_bce_adj,
        "losses_bce_dir": losses_bce_dir,
        "losses_mse": losses_mse,
    }
    json.dump(
        losses,
        open(f"models/{model_dir_name}_graph_size_{num_nodes}/losses.json", "w"),
    )


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # "train" or "test"
    mode = "test"
    # Name of model if training is selected it is created in testing it is loaded
    model_dir_name = "model_27"
    # Size of the graph you want to train or test on
    data_graph_size = 10

    # Hyperparameters only relevant if training, in testing they are loaded from json
    learning_rate = 0.001
    epochs = 2000
    learning_rate_steps = [epochs // 2, epochs // 5 * 4]
    batch_size = 10
    hidden_size_1 = 64
    hidden_size_2 = 16
    num_layers = 4
    lambda_ratios = {"kl_adj": 0.50, "kl_dir": 0.40, "l1": 0.10}

    sample_size = 0  # automatically set

    if mode == "test":
        dataset = Dataset(data_graph_size, test=True)
        data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_rnn(device, data_graph_size, model_dir_name, data)
        for i in range(len(dataset)):
            test_inference_rnn(
                device,
                hidden_size_1,
                hidden_size_2,
                data_graph_size,
                i,
                num_layers,
                model_dir_name,
                dataset,
            )
    if mode == "train":
        if not os.path.exists(f"models/{model_dir_name}_graph_size_{data_graph_size}"):
            os.mkdir(f"models/{model_dir_name}_graph_size_{data_graph_size}")

        dataset = Dataset(data_graph_size, test=False)
        sample_size = len(dataset)
        data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        rnn_graph = RNN(
            data_graph_size * 9,
            data_graph_size * 4,
            hidden_size_1,
            num_layers,
            output_size=hidden_size_2,
        ).to(device)
        rnn_edge = RNN(
            11,
            16,
            hidden_size_2,
            num_layers,
            output_size=7,
        ).to(device)
        train(
            rnn_graph,
            rnn_edge,
            device,
            learning_rate,
            learning_rate_steps,
            lambda_ratios,
            epochs,
            data_graph_size,
            data,
            model_dir_name,
        )
        hyperparameters = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "learning_rate_steps": learning_rate_steps,
            "batch_size": batch_size,
            "hidden_size_1": hidden_size_1,
            "hidden_size_2": hidden_size_2,
            "num_layers": num_layers,
            "data_graph_size": data_graph_size,
            "lambda_ratios": lambda_ratios,
            "sample_size": sample_size,
        }
        with open(
            f"models/{model_dir_name}_graph_size_{data_graph_size}/hyperparameters.json",
            "w",
        ) as f:
            json.dump(hyperparameters, f)
