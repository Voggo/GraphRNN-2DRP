import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import os
import json

from data import Dataset
from plot_rects import plot_rects
from generator import convert_graph_to_rects
from dataclasses_rect_point import Rectangle, Point
from utils import convert_center_to_lower_left

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        has_output=False,
        output_size=1,
        output_hidden_size=16,
    ):
        self.has_output = has_output
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        if has_output:
            self.output = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, output_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(output_hidden_size, output_size),
            )
        self.hidden = None

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size)

    def forward(self, x):
        output, self.hidden = self.rnn(x, self.hidden)
        if self.has_output:
            output = self.output(output)
        return output


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
        optimizer_rnn_graph, learning_rate_steps, gamma=0.1
    )
    scheduler_rnn_edge = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn_edge, learning_rate_steps, gamma=0.1
    )
    rnn_graph.train()
    rnn_edge.train()
    losses = []
    losses_bce_adj = []
    losses_bce_dir = []
    losses_mse = []
    for epoch in range(epochs):
        loss_sum = 0
        loss_sum_adj = 0
        loss_sum_dir = 0
        loss_sum_l2 = 0
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
            output_graph = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), num_nodes, num_nodes * 7).to(device)
            for i in range(num_nodes):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0)).to(device)
                rnn_edge.hidden[0, :, :] = output_graph[:, i, :].to(device)
                edge_input = x_bumpy[:, :, :, i].transpose(-2, -1).to(device)
                # (batch_size, seq_len(node_len), features)
                output_edge = rnn_edge(edge_input)
                output_edge = output_edge[:, :-1, :]
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
            y_pred_adj = y_pred[:, :, :num_nodes].clone()
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            loss_kl_adj = F.binary_cross_entropy(
                y_pred[:, :, :num_nodes], y[:, :, :num_nodes]
            )
            loss_kl_dir = F.binary_cross_entropy(
                y_pred[:, :, num_nodes : num_nodes * 6],
                y[:, :, num_nodes : num_nodes * 6],
            )
            loss_l2 = F.mse_loss(
                y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 6 :]
            )
            loss = (
                lambda_ratios["kl_adj"] * loss_kl_adj
                + lambda_ratios["kl_dir"] * loss_kl_dir
                + lambda_ratios["l2"] * loss_l2
            )
            loss_sum += loss.item()
            loss_sum_adj += loss_kl_adj.item() * lambda_ratios["kl_adj"]
            loss_sum_dir += loss_kl_dir.item() * lambda_ratios["kl_dir"]
            loss_sum_l2 += loss_l2.item() * lambda_ratios["l2"]
            loss.backward()
            optimizer_rnn_graph.step()
            optimizer_rnn_edge.step()
        scheduler_rnn_graph.step()
        scheduler_rnn_edge.step()
        batches = len(training_data) / batch_size
        losses.append(loss_sum / batches)
        losses_bce_adj.append(loss_sum_adj / batches)
        losses_bce_dir.append(loss_sum_dir / batches)
        losses_mse.append(loss_sum_l2 / batches)
        if epoch % 5 == 0:
            print(
                f"epoch: {epoch}, loss: {losses[-1]}, bce adj: {losses_bce_adj[-1]}, bce dir: {losses_bce_dir[-1]}, mse: {losses_mse[-1]}"
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
    return rnn_graph, rnn_edge, losses, losses_bce_adj, losses_bce_dir, losses_mse


def test_rnn(device, num_nodes, model_dir_name, test_data):
    hp = {}
    with open(
        f"models/{model_dir_name}_graph_size_{num_nodes}/hyperparameters.json", "r"
    ) as f:
        hp = json.load(f)
    lambda_ratios = hp["lambda_ratios"]
    rnn_graph = RNN((9 * num_nodes), hp["hidden_size_1"], hp["num_layers"]).to(device)
    rnn_edge = RNN(
        11, hp["hidden_size_2"], hp["num_layers"], has_output=True, output_size=7
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
        losses_mse = []
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
            output_graph = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), num_nodes, num_nodes * 7).to(device)
            for i in range(num_nodes):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0)).to(device)
                rnn_edge.hidden[0, :, :] = output_graph[:, i, :].to(device)
                edge_input = x_bumpy[:, :, :, i].transpose(-2, -1).to(device)
                # (batch_size, seq_len(node_len), features)
                output_edge = rnn_edge(edge_input)
                output_edge = output_edge[:, :-1, :]
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
            y_pred_adj = y_pred[:, :, :num_nodes].clone()
            y_pred[:, :, num_nodes * 6 :] *= y_pred_adj
            y_pred[:, :, num_nodes * 5 : num_nodes * 6] *= y_pred_adj
            y_pred[:, :, num_nodes * 4 : num_nodes * 5] *= y_pred_adj
            y_pred[:, :, num_nodes * 3 : num_nodes * 4] *= y_pred_adj
            y_pred[:, :, num_nodes * 2 : num_nodes * 3] *= y_pred_adj
            loss_kl_adj = F.binary_cross_entropy(
                y_pred[:, :, :num_nodes], y[:, :, :num_nodes]
            )
            loss_kl_dir = F.binary_cross_entropy(
                y_pred[:, :, num_nodes : num_nodes * 6],
                y[:, :, num_nodes : num_nodes * 6],
            )
            loss_l2 = F.mse_loss(
                y_pred[:, :, num_nodes * 6 :], y[:, :, num_nodes * 6 :]
            )
            loss = (
                lambda_ratios["kl_adj"] * loss_kl_adj
                + lambda_ratios["kl_dir"] * loss_kl_dir
                + lambda_ratios["l2"] * loss_l2
            )
            losses.append(loss.item())
            losses_bce_adj.append(loss_kl_adj.item())
            losses_bce_dir.append(loss_kl_dir.item())
            losses_mse.append(loss_l2.item())
            print(
                f"loss: {loss.item()}, bce adj: {loss_kl_adj.item()}, bce dir: {loss_kl_dir.item()}, mse: {loss_l2.item()}"
            )
        losses_mean = sum(losses) / len(losses)
        losses_bce_adj_mean = sum(losses_bce_adj) / len(losses_bce_adj)
        losses_bce_dir_mean = sum(losses_bce_dir) / len(losses_bce_dir)
        losses_mse_mean = sum(losses_mse) / len(losses_mse)
        print(
            f"mean loss: {losses_mean}",
            f"mean bce adj: {losses_bce_adj_mean}, bce dir: {losses_bce_dir_mean}, mse: {losses_mse_mean}",
        )
        losses = {
            "losses": losses,
            "losses_bce_adj": losses_bce_adj,
            "losses_bce_dir": losses_bce_dir,
            "losses_mse": losses_mse,
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
    batch_size=1,
):
    rnn_graph = RNN((num_nodes * 9), hidden_size_1, num_layers).to(device)
    rnn_edge = RNN(11, hidden_size_2, num_layers, has_output=True, output_size=7).to(
        device
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
    with torch.no_grad():
        rnn_graph.eval()
        rnn_edge.eval()
        data = Dataset(num_nodes, test=False)
        print(data.data_bfs_adj[graph])
        print(data.data_bfs_edge_dir[graph])
        print(data.data_bfs_offset[graph])
        nodes = data.data_bfs_nodes[graph]
        x_step = torch.ones(batch_size, 1, num_nodes * 9).to(device)
        y_pred = torch.zeros(batch_size, num_nodes, num_nodes * 7).to(device)
        for i in range(num_nodes):
            output_graph = rnn_graph(x_step)
            rnn_edge.hidden = rnn_edge.init_hidden(batch_size).to(device)
            rnn_edge.hidden[0, :, :] = output_graph[:, -1, :]
            edge_input_step = torch.ones(batch_size, 11).to(device)
            edge_y_pred = torch.zeros(batch_size, num_nodes, 9).to(device)
            for j in range(i + 1):
                output_edge = rnn_edge(edge_input_step.unsqueeze(0))
                output_edge[:, 0, 0:1] = torch.bernoulli(
                    F.sigmoid(output_edge[:, 0, 0:1])
                )
                dir = torch.multinomial(F.sigmoid(output_edge[:, 0, 1:6]), 1)
                output_edge[:, 0, 1 + dir] = 1
                output_edge[:, 0, 1 : dir + 1] = 0
                output_edge[:, 0, dir + 2 : 6] = 0
                edge_input_step[:, :7] = output_edge[0, 0, :]
                if j < len(nodes):
                    edge_input_step[0, 7] = nodes[j][0]
                    edge_input_step[0, 8] = nodes[j][1]
                else:
                    edge_input_step[0, 7] = 0
                    edge_input_step[0, 8] = 0
                if i < len(nodes):
                    edge_input_step[0, 9] = nodes[i][0]
                    edge_input_step[0, 10] = nodes[i][1]
                else:
                    edge_input_step[0, 9] = 0
                    edge_input_step[0, 10] = 0
                edge_y_pred[:, j, :7] = output_edge[:, :, :7]
                edge_y_pred[:, j, 7:] = edge_input_step[:, 7:9]
            edge_y_pred = edge_y_pred.transpose(-2, -1).flatten(start_dim=1, end_dim=2)
            y_pred[:, i, :] = edge_y_pred[:, : num_nodes * 7]
            x_step = edge_y_pred.unsqueeze(0).to(device)

        adj = y_pred[0, :, :num_nodes].reshape(num_nodes, num_nodes).to(torch.int64)
        adj.diagonal().fill_(0)
        adj = adj + adj.T
        edge_dir = y_pred[0, :, num_nodes : num_nodes * 6]
        edge_dir_splits = torch.split(edge_dir, edge_dir.size(1) // 5, dim=1)
        edge_dir = torch.stack(edge_dir_splits, dim=2)
        edge_dir = torch.argmax(edge_dir, dim=2)

        edge_dir.diagonal().fill_(0)
        mapping = torch.tensor([0, 3, 4, 1, 2])
        edge_dir = edge_dir + mapping[edge_dir].T
        offset = y_pred[0, :, num_nodes * 6 :].reshape(num_nodes, num_nodes)
        offset.diagonal().fill_(0)
        offset = offset + offset.T
        nodes = [Rectangle(node[0], node[1], 0) for node in nodes]
        rects = convert_graph_to_rects(nodes, adj, edge_dir, offset)
        rects = convert_center_to_lower_left(rects)
        plot_rects(
            rects,
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
        os.path.isfile(f"models/rnn_graph_model_{num_nodes}.pth")
        or os.path.isfile(f"models/rnn_edge_model_{num_nodes}.pth")
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
    model_dir_name = "model_3"

    # Hyperparameters only relevant if training, in testing they are loaded from json
    learning_rate = 0.001
    epochs = 10000
    learning_rate_steps = [epochs // 3, epochs // 5 * 4]
    batch_size = 1
    hidden_size_1 = 64
    hidden_size_2 = 64
    num_layers = 4
    data_graph_size = 6
    lambda_ratios = {"kl_adj": 0.20, "kl_dir": 0.50, "l2": 0.30}

    if mode == "test":
        dataset = Dataset(data_graph_size, test=True)
        data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_rnn(device, data_graph_size, model_dir_name, data)
        for i in range(5):
            test_inference_rnn(
                device,
                hidden_size_1,
                hidden_size_2,
                data_graph_size,
                i,
                num_layers,
                model_dir_name,
            )
    if mode == "train":
        if not os.path.exists(f"models/{model_dir_name}_graph_size_{data_graph_size}"):
            os.mkdir(f"models/{model_dir_name}_graph_size_{data_graph_size}")

        dataset = Dataset(data_graph_size, test=False)
        data = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        rnn_graph = RNN((data_graph_size * 9), hidden_size_1, num_layers).to(device)
        rnn_edge = RNN(
            11, hidden_size_2, num_layers, has_output=True, output_size=7
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
        }
        with open(
            f"models/{model_dir_name}_graph_size_{data_graph_size}/hyperparameters.json",
            "w",
        ) as f:
            json.dump(hyperparameters, f)
