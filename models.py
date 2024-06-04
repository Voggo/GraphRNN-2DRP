import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn.utils import clip_grad_norm_
from data import Dataset
from plot_rects import plot_rects
from generator import (
    convert_graph_to_rects,
    generate_random_graph,
    generate_random_rect_positions,
)
from dataclasses_rect_point import Rectangle
from utils import sample_graph, accuracy
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
        encoded_input = self.embedding(x)
        encoded_input = self.relu(encoded_input)
        output, self.hidden = self.rnn(encoded_input, self.hidden)
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
    test_dataset = Dataset(num_nodes, test=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    rnn_graph.train()
    rnn_edge.train()
    losses = []
    losses_bce_adj = []
    losses_bce_dir = []
    losses_mae = []
    test_mean_loss = []
    test_mean_accuracy = []
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
                output_edge[:, :, 0:1] = F.sigmoid(output_edge[:, :, 0:1])
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
            y_pred_adj = torch.round(torch.tril(y_pred[:, :, :num_nodes].clone()))
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            y_pred[:, :, num_nodes : num_nodes * 2] *= torch.tril(1 - y_pred_adj)
            y_pred[:, :, :num_nodes] = torch.tril(y_pred[:, :, :num_nodes].clone())
            loss_kl_adj = (
                F.binary_cross_entropy(y_pred[:, :, :num_nodes], y[:, :, :num_nodes])
                / batch_size
            )
            splits = torch.split(
                y_pred[:, :, num_nodes : num_nodes * 6], num_nodes, dim=2
            )
            y_pred_dir = torch.stack(splits, dim=1)
            loss_kl_dir = (
                F.cross_entropy(
                    y_pred_dir,
                    y_bumpy[:, 1, :, :].to(torch.int64),
                )
                / batch_size
            )
            loss_l1 = (
                F.l1_loss(y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 2 :])
                / batch_size
            )
            loss = (
                lambda_ratios["kl_adj"] * loss_kl_adj
                + lambda_ratios["kl_dir"] * loss_kl_dir
                + lambda_ratios["l1"] * loss_l1
            )
            loss_sum += loss_kl_adj.item() + loss_kl_dir.item() + loss_l1.item()
            loss_sum_adj += loss_kl_adj.item()
            loss_sum_dir += loss_kl_dir.item()
            loss_sum_l1 += loss_l1.item()
            loss.backward()
            clip_grad_norm_(rnn_graph.parameters(), 0.01)
            clip_grad_norm_(rnn_edge.parameters(), 0.01)
            # grad_norm_graph = rnn_graph.embedding.weight.grad.norm()
            # grad_norm_edge = rnn_edge.embedding.weight.grad.norm()
            # print(f"grad norm graph: {grad_norm_graph}")
            # print(f"grad norm edge:  {grad_norm_edge}")
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
                f"epoch: {epoch}, loss: {losses[-1]}, bce adj: {losses_bce_adj[-1]}, bce dir: {losses_bce_dir[-1]}, mae: {losses_mae[-1]}"  # pylint: disable=line-too-long
            )
        if (epoch + 1) % 50 == 0 and not epoch == 0:
            torch.save(
                rnn_graph.state_dict(),
                f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
            )
            torch.save(
                rnn_edge.state_dict(),
                f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
            )
            metrics = test_rnn(device, rnn_graph, rnn_edge, num_nodes, test_data)
            test_mean_accuracy.append(metrics["test_mean_accuracy"])
            test_mean_loss.append(metrics["test_mean_loss"])
            rnn_edge.train()
            rnn_graph.train()

    torch.save(
        rnn_graph.state_dict(),
        f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_graph_model.pth",
    )
    torch.save(
        rnn_edge.state_dict(),
        f"models/{model_dir_name}_graph_size_{num_nodes}/rnn_edge_model.pth",
    )
    return (
        rnn_graph,
        rnn_edge,
        losses,
        losses_bce_adj,
        losses_bce_dir,
        losses_mae,
        test_mean_loss,
        test_mean_accuracy,
    )


def test_rnn(device, rnn_graph, rnn_edge, num_nodes, test_data):
    rnn_graph.eval()
    rnn_edge.eval()
    batch_size = test_data.batch_size
    with torch.no_grad():
        # losses pr graph in the dataset
        metrics = {
            "test_losses": {"loss": [], "bce_adj": [], "bce_dir": [], "mae": []},
            "test_accuracies": {
                "adj": [],
                "dir": [],
            },
        }
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
                output_edge[:, :, 0:1] = F.sigmoid(output_edge[:, :, 0:1])
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
            y_pred_adj = torch.round(torch.tril(y_pred[:, :, :num_nodes].clone()))
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            y_pred[:, :, num_nodes : num_nodes * 2] *= torch.tril(1 - y_pred_adj)
            y_pred[:, :, :num_nodes] = torch.tril(y_pred[:, :, :num_nodes].clone())
            loss_kl_adj = (
                F.binary_cross_entropy(y_pred[:, :, :num_nodes], y[:, :, :num_nodes])
                / batch_size
            )
            splits = torch.split(
                y_pred[:, :, num_nodes : num_nodes * 6], num_nodes, dim=2
            )
            y_pred_dir = torch.stack(splits, dim=1)
            loss_kl_dir = (
                F.cross_entropy(
                    y_pred_dir,
                    y_bumpy[:, 1, :, :].to(torch.int64),
                )
                / batch_size
            )
            loss_l1 = (
                F.l1_loss(y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 2 :])
                / batch_size
            )
            loss = loss_kl_adj + loss_kl_dir + loss_l1
            metrics["test_losses"]["loss"].append(loss.item())
            metrics["test_losses"]["bce_adj"].append(loss_kl_adj.item())
            metrics["test_losses"]["bce_dir"].append(loss_kl_dir.item())
            metrics["test_losses"]["mae"].append(loss_l1.item())
            print(
                f"loss: {loss.item()}, bce adj: {loss_kl_adj.item()}, bce dir: {loss_kl_dir.item()}, mae: {loss_l1.item()}"  # pylint: disable=line-too-long
            )
            # number of cells predicted is only lower triangle
            cells = 0.5 * (num_nodes - 1) * num_nodes * batch_size
            accuracies = accuracy(y_pred, y, cells, num_nodes)
            metrics["test_accuracies"]["adj"].append(accuracies["accuracy_adj"])
            metrics["test_accuracies"]["dir"].append(accuracies["accuracy_dir"])
            print(
                f"accuracy adj: {accuracies['accuracy_adj']}, accuracy dir: {accuracies['accuracy_dir']}"  # pylint: disable=line-too-long
            )
        losses = metrics["test_losses"]
        metrics["test_mean_loss"] = {
            "loss_mean": sum(losses["loss"]) / len(losses["loss"]),
            "loss_bce_adj_mean": sum(losses["bce_adj"]) / len(losses["bce_adj"]),
            "loss_bce_dir_mean": sum(losses["bce_dir"]) / len(losses["bce_dir"]),
            "loss_mse_mean": sum(losses["mae"]) / len(losses["mae"]),
        }
        accuracies = metrics["test_accuracies"]
        metrics["test_mean_accuracy"] = {
            "accuracy_adj": sum(accuracies["adj"]) / len(accuracies["adj"]),
            "accuracy_dir": sum(accuracies["dir"]) / len(accuracies["dir"]),
        }
        print(
            f"mean loss: {metrics['test_mean_loss']['loss_mean']}, mean bce adj: {metrics['test_mean_loss']['loss_bce_adj_mean']}, mean bce dir: {metrics['test_mean_loss']['loss_bce_dir_mean']}, mean mae: {metrics['test_mean_loss']['loss_mse_mean']}"  # pylint: disable=line-too-long
        )
        print(
            f"mean accuracy adj: {metrics['test_mean_accuracy']['accuracy_adj']}, mean accuracy dir: {metrics['test_mean_accuracy']['accuracy_dir']}"  # pylint: disable=line-too-long
        )
        return metrics


def test_inference_rnn(
    device,
    input_embedding_scalar,
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
        num_nodes * input_embedding_scalar,
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
            if i < len(nodes) - 1:
                edge_input_step[:, 0, 9] = nodes[i + 1][0]
                edge_input_step[:, 0, 10] = nodes[i + 1][1]
            else:
                edge_input_step[:, 0, 9] = 0
                edge_input_step[:, 0, 10] = 0
            edge_y_pred = torch.zeros(batch_size, num_nodes, 9).to(device)
            for j in range(i + 1):
                edge_input_step_all[:, j, :] = edge_input_step
                _, output_edge = rnn_edge(edge_input_step)
                output_edge[:, 0, 0:1] = torch.bernoulli(
                    F.sigmoid(output_edge[:, 0, 0:1])
                )
                direction = 0
                if output_edge[:, 0, 0] == 0:
                    probabilities = F.softmax(output_edge[:, 0, 1:6], dim=1)
                    direction = torch.multinomial(probabilities, 1).squeeze()
                else:
                    probabilities = F.softmax(output_edge[:, 0, 2:6], dim=1)
                    direction = torch.multinomial(probabilities, 1).squeeze() + 1
                output_edge[:, 0, 1 + direction] = 1
                output_edge[:, 0, 1 : direction + 1] = 0
                output_edge[:, 0, direction + 2 : 6] = 0
                output_edge[:, 0, 0:] *= output_edge[:, 0, 0:1].clone()
                edge_input_step[:, :, :7] = output_edge[0, 0, :]
                if j < len(nodes) - 1:
                    edge_input_step[:, 0, 7] = nodes[j + 1][0]
                    edge_input_step[:, 0, 8] = nodes[j + 1][1]
                else:
                    edge_input_step[:, 0, 7] = 0
                    edge_input_step[:, 0, 8] = 0
                if i < len(nodes) - 1:
                    edge_input_step[:, 0, 9] = nodes[i + 1][0]
                    edge_input_step[:, 0, 10] = nodes[i + 1][1]
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
        y_pred = torch.cat(
            (torch.zeros((1, 1, num_nodes * 7)).to(device), y_pred), dim=1
        ).to(device)
        adj = y_pred[0, :, :num_nodes].reshape(num_nodes, num_nodes).to(torch.int64)
        adj.diagonal().fill_(0)
        adj = adj + adj.T.to(torch.int64)
        edge_dir = y_pred[0, :, num_nodes : num_nodes * 6]
        edge_dir_splits = torch.split(edge_dir, edge_dir.size(1) // 5, dim=1)
        edge_dir = torch.stack(edge_dir_splits, dim=2)
        edge_dir = torch.argmax(edge_dir, dim=2)
        edge_dir.diagonal().fill_(0)
        mapping = torch.tensor([0, 3, 4, 1, 2]).to(device)
        edge_dir = (edge_dir + mapping[edge_dir].T).to(torch.int64)
        offset = y_pred[0, :, num_nodes * 6 :].reshape(num_nodes, num_nodes)
        offset.diagonal().fill_(0)
        offset = offset + offset.T
        # print("current adj:")
        # print(data.data_bfs_adj[graph])
        # print(adj)
        # print("current edge dir:")
        # print(data.data_bfs_edge_dir[graph])
        # print(edge_dir)
        # print("current offset:")
        # print(data.data_bfs_offset[graph])
        # print(offset)
        # Convert edge_dir to boolean tensor where True indicates the presence of an edge
        edge_mask = edge_dir > 0
        # Use the mask to filter the adjacency matrix
        adj = torch.where(edge_mask, adj, torch.zeros_like(adj))
        best_rects = None
        max_utility = 0
        for _ in range(1):
            nodes_rects = [
                Rectangle(node[0].item(), node[1].item(), 0) for node in nodes
            ]
            # sampled_graph = sample_graph(adj.cpu().numpy())
            sampled_graph = adj.numpy()
            rects = convert_graph_to_rects(
                nodes_rects, sampled_graph, edge_dir.cpu().numpy(), offset.cpu().numpy()
            )
            # rects = convert_center_to_lower_left(rects)
            fill_ratio, overlap_area, cutoff_area = evaluate_solution(rects, 100, 100)
            utility = fill_ratio - 0 * overlap_area / 10000 - 0 * cutoff_area / 10000
            # print(f"utility: {utility}")
            if utility > max_utility:
                best_rects = rects
        # print(f"max fill ratio: {max_fill_ratio}")
        # print(f"min overlap area: {min_overlap_area}")
        # print(f"cut off area: {cutoff_area}")
        # plot_rects(
        #     best_rects,
        #     ax_lim=100,
        #     ay_lim=100,
        #     ax_min=-50,
        #     ay_min=-50,
        #     filename="rnn_rnn.png",
        # )
        return best_rects


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
    axs[1, 1].set_title("MAE Loss", fontsize=10)
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
    (
        rnn_graph,
        rnn_edge,
        losses,
        losses_bce_adj,
        losses_bce_dir,
        losses_mse,
        test_mean_loss,
        test_mean_accuracy,
    ) = train_rnn(
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
    test_metrics = {
        "test_mean_loss": test_mean_loss,
        "test_mean_accuracy": test_mean_accuracy,
    }
    json.dump(
        test_metrics,
        open(f"models/{model_dir_name}_graph_size_{num_nodes}/test_metrics.json", "w"),
    )


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = False

    # "train" or "test"
    mode = "test"
    # Name of model if training is selected it is created in testing it is loaded
    model_dir_name = "model_14"
    # Size of the graph you want to train or test on
    data_graph_size = 16

    # Hyperparameters only relevant if training, in testing they are loaded from json
    learning_rate = 0.001
    epochs = 4000
    learning_rate_steps = [epochs // 1.5, epochs // 5 * 4]
    batch_size = 50
    input_embedding_scalar = 6  # divided by 9 to get ratio of input
    hidden_size_1 = 64
    hidden_size_2 = 32
    num_layers = 5
    lambda_ratios = {"kl_adj": 0.10, "kl_dir": 0.60, "l1": 0.30}

    sample_size = 0  # automatically set

    if mode == "test":
        test = True
        dataset = Dataset(data_graph_size, test=test)
        num_samples = 100  # specify the number of samples you want
        limited_dataset = torch.utils.data.Subset(dataset, range(num_samples))
        data = torch.utils.data.DataLoader(
            limited_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        hp = {}
        with open(  # pylint: disable=unspecified-encoding
            f"models/{model_dir_name}_graph_size_{data_graph_size}/hyperparameters.json",
            "r",
        ) as f:
            hp = json.load(f)
        rnn_graph = RNN(
            9 * data_graph_size,
            hp["input_embedding_scalar"] * data_graph_size,
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
        ).to(device)
        rnn_graph.load_state_dict(
            torch.load(
                f"models/{model_dir_name}_graph_size_{data_graph_size}/rnn_graph_model.pth",
                map_location=torch.device("cpu") if not use_cuda else None,
            )
        )
        rnn_edge.load_state_dict(
            torch.load(
                f"models/{model_dir_name}_graph_size_{data_graph_size}/rnn_edge_model.pth",
                map_location=torch.device("cpu") if not use_cuda else None,
            )
        )
        # test_metrics = test_rnn(device, rnn_graph, rnn_edge, data_graph_size, data)
        fill_ratios = []
        overlap_areas = []
        cutoff_areas = []
        test_size = len(limited_dataset)
        data_type = "test_data" if test else "training_data"
        if not os.path.exists(
            f"models/{model_dir_name}_graph_size_{data_graph_size}/packing_solutions_{data_type}"
        ):
            os.mkdir(
                f"models/{model_dir_name}_graph_size_{data_graph_size}/packing_solutions_{data_type}"
            )

        for i in range(test_size):
            print(f"graph: {i}")
            max_utility = 0
            best_rects = None
            for _ in range(200):
                rects = test_inference_rnn(
                    device,
                    hp["input_embedding_scalar"],
                    hp["hidden_size_1"],
                    hp["hidden_size_2"],
                    hp["data_graph_size"],
                    i,
                    hp["num_layers"],
                    model_dir_name,
                    dataset,
                )
                fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                    rects, 100, 100
                )
                utility = (
                    fill_ratio
                    - 0.0 * (overlap_area / 10000)
                    - 0.0 * (cutoff_area / 10000)
                )
                if utility > max_utility:
                    max_utility = utility
                    best_rects = rects
            fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                best_rects, 100, 100
            )
            fill_ratios.append(fill_ratio)
            overlap_areas.append(overlap_area)
            cutoff_areas.append(cutoff_area)
            print(
                f"fill ratio: {fill_ratio}, overlap area: {overlap_area}, cutoff area: {cutoff_area}"
            )  # pylint: disable=line-too-long
            plot_rects(
                best_rects,
                ax_lim=100,
                ay_lim=100,
                ax_min=-50,
                ay_min=-50,
                filename=f"models/{model_dir_name}_graph_size_{data_graph_size}/packing_solutions_{data_type}/graph_{i}.png",  # pylint: disable=line-too-long
                show=False,
                title=f"$F={round(fill_ratio * 100, 1)}\% \quad O = {round(overlap_area, 1)}\quad C = {round(cutoff_area, 1)}$",  # pylint: disable=line-too-long
            )
        avr_fill_ratio = sum(fill_ratios) / test_size
        avr_overlap_area = sum(overlap_areas) / test_size
        avr_cutoff_area = sum(cutoff_areas) / test_size
        print(f"avr fill ratio: {avr_fill_ratio}")
        print(f"avr overlap area: {avr_overlap_area}")
        print(f"avr cutoff area: {avr_cutoff_area}")
        json.dump(
            {
                "fill_ratios": fill_ratios,
                "overlap_areas": overlap_areas,
                "cutoff_areas": cutoff_areas,
                "avr_fill_ratio": avr_fill_ratio,
                "avr_overlap_area": avr_overlap_area,
                "avr_cutoff_area": avr_cutoff_area,
            },
            open(
                f"models/{model_dir_name}_graph_size_{data_graph_size}/test_inference_stats_{data_type}.json",  # pylint: disable=line-too-long
                "w",
            ),
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
            data_graph_size * input_embedding_scalar,
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
            "input_embedding_scalar": input_embedding_scalar,
            "hidden_size_1": hidden_size_1,
            "hidden_size_2": hidden_size_2,
            "num_layers": num_layers,
            "data_graph_size": data_graph_size,
            "lambda_ratios": lambda_ratios,
            "sample_size": sample_size,
        }
        json.dump(
            hyperparameters,
            open(
                f"models/{model_dir_name}_graph_size_{data_graph_size}/hyperparameters.json",
                "w",
            ),
        )
    if mode == "random_graph":
        dataset = Dataset(data_graph_size, test=True)
        num_samples = 100  # specify the number of samples you want
        limited_dataset = torch.utils.data.Subset(dataset, range(num_samples))
        data = torch.utils.data.DataLoader(
            limited_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        fill_ratios = []
        overlap_areas = []
        cutoff_areas = []
        test_size = len(limited_dataset)
        for i in range(test_size):
            nodes = dataset.data_bfs_nodes[i]
            rects = []
            best_rects = None
            max_fill_ratio = 0
            for _ in range(200):
                rects = []
                for node in nodes:
                    rects.append(Rectangle(node[0], node[1], 0))
                adj, edge_dir, offset = generate_random_graph(rects, 100, 100)
                new_rects = convert_graph_to_rects(rects, adj, edge_dir, offset)
                fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                    new_rects, 100, 100
                )
                if fill_ratio > max_fill_ratio:
                    max_fill_ratio = fill_ratio
                    best_rects = new_rects

            fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                best_rects, 100, 100
            )
            print(f"fill ratio: {fill_ratio}")
            print(f"overlap area: {overlap_area}")
            print(f"cutoff area: {cutoff_area}")
            fill_ratios.append(fill_ratio)
            overlap_areas.append(overlap_area)
            cutoff_areas.append(cutoff_area)
            # plot_rects(rects, ax_lim=100, ay_lim=100, ax_min=-50, ay_min=-50)
        avr_fill_ratio = sum(fill_ratios) / test_size
        avr_overlap_area = sum(overlap_areas) / test_size
        avr_cutoff_area = sum(cutoff_areas) / test_size
        print(f"avr fill ratio: {avr_fill_ratio}")
        print(f"avr overlap area: {avr_overlap_area}")
        print(f"avr cutoff area: {avr_cutoff_area}")
        json.dump(
            {
                "fill_ratios": fill_ratios,
                "overlap_areas": overlap_areas,
                "cutoff_areas": cutoff_areas,
                "avr_fill_ratio": avr_fill_ratio,
                "avr_overlap_area": avr_overlap_area,
                "avr_cutoff_area": avr_cutoff_area,
            },
            open("datasets/random_graph_stats.json", "w"),
        )

    if mode == "random_pos":
        dataset = Dataset(data_graph_size, test=True)
        num_samples = 100  # specify the number of samples you want
        limited_dataset = torch.utils.data.Subset(dataset, range(num_samples))
        data = torch.utils.data.DataLoader(
            limited_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        fill_ratios = []
        overlap_areas = []
        cutoff_areas = []
        test_size = len(limited_dataset)
        for i in range(test_size):
            nodes = dataset.data_bfs_nodes[i]
            rects = []
            best_rects = None
            max_fill_ratio = 0
            for _ in range(200):
                rects = []
                for node in nodes:
                    rects.append(Rectangle(node[0], node[1], 0))
                rects = generate_random_rect_positions(rects, 100, 100)
                fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                    rects, 100, 100
                )
                if fill_ratio > max_fill_ratio:
                    max_fill_ratio = fill_ratio
                    best_rects = rects
            fill_ratio, overlap_area, cutoff_area = evaluate_solution(
                best_rects, 100, 100
            )
            print(f"fill ratio: {fill_ratio}")
            print(f"overlap area: {overlap_area}")
            print(f"cutoff area: {cutoff_area}")
            fill_ratios.append(fill_ratio)
            overlap_areas.append(overlap_area)
            cutoff_areas.append(cutoff_area)
            # plot_rects(rects, ax_lim=150, ay_lim=150, ax_min=0, ay_min=0)
        avr_fill_ratio = sum(fill_ratios) / test_size
        avr_overlap_area = sum(overlap_areas) / test_size
        avr_cutoff_area = sum(cutoff_areas) / test_size
        print(f"avr fill ratio: {avr_fill_ratio}")
        print(f"avr overlap area: {avr_overlap_area}")
        print(f"avr cutoff area: {avr_cutoff_area}")
        json.dump(
            {
                "fill_ratios": fill_ratios,
                "overlap_areas": overlap_areas,
                "cutoff_areas": cutoff_areas,
                "avr_fill_ratio": avr_fill_ratio,
                "avr_overlap_area": avr_overlap_area,
                "avr_cutoff_area": avr_cutoff_area,
            },
            open("datasets/random_pos_stats.json", "w"),
        )
