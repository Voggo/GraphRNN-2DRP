import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
from data import DatasetSimple, Dataset
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


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


def train_rnn_mlp(
    rnn, mlp, device, learning_rate, epochs, max_num_nodes, training_data
):
    optimizer_rnn = torch.optim.Adam(list(rnn.parameters()), lr=learning_rate)
    optimizer_mlp = torch.optim.Adam(list(mlp.parameters()), lr=learning_rate)
    loss = 0
    rnn.train()
    mlp.train()
    loss_sum = 0
    for epoch in range(epochs):
        for batch in training_data:
            rnn.zero_grad()
            mlp.zero_grad()
            x = batch["x"].float().to(device)
            y = batch["y"].float().to(device)
            rnn.hidden = rnn.init_hidden(x.size(0))
            output = rnn(x)
            output = mlp(output)
            y_pred = F.sigmoid(output)
            loss = F.binary_cross_entropy(y_pred, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer_rnn.step()
            optimizer_mlp.step()
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")

    torch.save(rnn.state_dict(), f"models/rnn_model_{max_num_nodes}.pth")
    torch.save(mlp.state_dict(), f"models/mlp_model_{max_num_nodes}.pth")

    return rnn, mlp, loss_sum


def test_rnn_mlp(device, training_data, max_num_nodes, rnn, mlp):
    rnn.load_state_dict(torch.load(f"models/rnn_model_{max_num_nodes}.pth"))
    mlp.load_state_dict(torch.load(f"models/mlp_model_{max_num_nodes}.pth"))
    rnn.eval()
    mlp.eval()
    loss_sum = 0
    for batch in training_data:
        x = batch["x"].float().to(device)
        y = batch["y"].float().to(device)
        rnn.hidden = rnn.init_hidden(x.size(0))
        output = rnn(x)
        output = mlp(output)
        y_pred = F.sigmoid(output)
        loss = F.binary_cross_entropy(y_pred, y)
        y_pred = torch.round(y_pred)
        print(y_pred)
        loss_sum += loss.item()
        correct_count = torch.sum(y_pred == y).item()
        correct_percent = (correct_count / y.numel()) * 100
        print(f"Correct: {correct_percent}%")


def test_inference_rnn_mlp(device, rnn, mlp, max_num_nodes, batch_size=1):
    rnn.load_state_dict(torch.load(f"models/rnn_model_{max_num_nodes}.pth"))
    mlp.load_state_dict(torch.load(f"models/mlp_model_{max_num_nodes}.pth"))
    rnn.eval()
    mlp.eval()
    rnn.hidden = rnn.init_hidden(batch_size)
    x_step = torch.ones(batch_size, 1, max_num_nodes)
    y_pred = torch.zeros(batch_size, max_num_nodes, max_num_nodes)
    for i in range(max_num_nodes):
        output = rnn(x_step)
        output = mlp(output)
        y_pred[:, i : i + 1, :] = torch.bernoulli(F.sigmoid(output))
        x_step = torch.round(y_pred[:, i : i + 1, :])
        rnn.hidden = rnn.hidden.data
    for i in range(batch_size):
        print(torch.round(y_pred[i]))


def train_rnn_rnn(
    rnn_graph,
    rnn_edge,
    device,
    learning_rate,
    epochs,
    max_num_nodes,
    training_data,
    lambda_kl_adj=0.30,
    lambda_kl_dir=0.60,
    lambda_l2=0.10,
):
    optimizer_rnn_graph = torch.optim.Adam(
        list(rnn_graph.parameters()), lr=learning_rate
    )
    optimizer_rnn_edge = torch.optim.Adam(list(rnn_edge.parameters()), lr=learning_rate)
    scheduler_rnn_graph = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn_graph, [epochs // 3, epochs // 4 * 3], gamma=0.1
    )
    scheduler_rnn_edge = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_rnn_edge, [epochs // 3, epochs // 4 * 3], gamma=0.1
    )
    rnn_graph.train()
    rnn_edge.train()
    for epoch in range(epochs):
        loss_sum = 0
        loss_bce_adj_sum = 0
        loss_bce_dir_sum = 0
        loss_mae_sum = 0
        for batch in training_data:
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
            )
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
            )
            rnn_graph.hidden = rnn_graph.init_hidden(x.size(0))
            output_graph = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), max_num_nodes, max_num_nodes * 7)
            for i in range(max_num_nodes):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0))
                rnn_edge.hidden[0, :, :] = output_graph[:, i, :]
                edge_input = x_bumpy[:, :, :, i].transpose(
                    -2, -1
                )  # (batch_size, seq_len(node_len), features)
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
            y_pred_adj = y_pred[:, :, :max_num_nodes].clone()
            y_pred[:, :, max_num_nodes * 6 :] *= y_pred_adj
            loss_kl_adj = F.binary_cross_entropy(
                y_pred[:, :, :max_num_nodes], y[:, :, :max_num_nodes]
            )
            loss_kl_dir = F.binary_cross_entropy(
                y_pred[:, :, max_num_nodes : max_num_nodes * 6],
                y[:, :, max_num_nodes : max_num_nodes * 6],
            )
            loss_l2 = F.l1_loss(
                y_pred[:, :, max_num_nodes * 6 :], y[:, :, max_num_nodes * 6 :]
            )
            loss = (
                lambda_kl_adj * loss_kl_adj
                + lambda_kl_dir * loss_kl_dir
                + lambda_l2 * loss_l2
            )
            loss_sum += loss.item()
            loss_bce_adj_sum += loss_kl_adj.item() * lambda_kl_adj
            loss_bce_dir_sum += loss_kl_dir.item() * lambda_kl_dir
            loss_mae_sum += loss_l2.item() * lambda_l2
            loss.backward()
            optimizer_rnn_graph.step()
            optimizer_rnn_edge.step()
        scheduler_rnn_graph.step()
        scheduler_rnn_edge.step()
        if epoch % 5 == 0:
            print(
                f"epoch: {epoch}, loss: {loss_sum/5}, bce adj: {loss_bce_adj_sum/5}, bce dir: {loss_bce_dir_sum/5}, mae: {loss_mae_sum/5}"
            )
        if epoch % 100 == 0 and not epoch == 0:
            torch.save(
                rnn_graph.state_dict(), f"models/rnn_graph_model_{max_num_nodes}.pth"
            )
            torch.save(
                rnn_edge.state_dict(), f"models/rnn_edge_model_{max_num_nodes}.pth"
            )

    torch.save(rnn_graph.state_dict(), f"models/rnn_graph_model_{max_num_nodes}.pth")
    torch.save(rnn_edge.state_dict(), f"models/rnn_edge_model_{max_num_nodes}.pth")
    return rnn_graph, rnn_edge, loss_sum


def test_rnn_rnn(device, training_data, max_num_nodes, rnn_graph, rnn_edge):
    rnn_graph.load_state_dict(torch.load(f"models/rnn_graph_model_{max_num_nodes}.pth"))
    rnn_edge.load_state_dict(torch.load(f"models/rnn_edge_model_{max_num_nodes}.pth"))
    rnn_graph.eval()
    rnn_edge.eval()
    loss_sum = 0
    with torch.no_grad():
        for batch in training_data:
            x_bumpy = batch["x"].float().to(device)
            y_bumpy = batch["y"].float().to(device)
            x = torch.flatten(x_bumpy, start_dim=1, end_dim=2).transpose(-2, -1)
            y = torch.flatten(y_bumpy, start_dim=1, end_dim=2).transpose(-2, -1)
            rnn_graph.hidden = rnn_graph.init_hidden(x.size(0))
            output_graph = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), max_num_nodes, max_num_nodes * 3)
            for i in range(max_num_nodes):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0))
                rnn_edge.hidden[0, :, :] = output_graph[:, i, :]
                edge_input = x_bumpy[:, :, 1:, i]
                output_edge = rnn_edge(
                    edge_input.transpose(-2, -1)
                )  # (batch_size, seq_len(node_len), features)
                output_edge[:, :, 0:1] = torch.bernoulli(
                    F.sigmoid(output_edge[:, :, 0:1])
                )
                output_edge = output_edge.transpose(-2, -1).flatten(
                    start_dim=1, end_dim=2
                )
                y_pred[:, i, :] = output_edge
            loss_k2 = F.binary_cross_entropy(
                y_pred[:, :, :max_num_nodes], y[:, :, :max_num_nodes]
            )
            loss_l2 = F.mae_loss(y_pred[:, :, max_num_nodes:], y[:, :, max_num_nodes:])
            loss = loss_k2 + loss_l2
            loss_sum += loss.item()
            print(f"loss: {loss.item()}")


def test_inference_rnn_rnn(
    device, hidden_size_1, hidden_size_2, max_num_nodes, graph, num_layers, batch_size=1
):
    rnn_graph = RNN((max_num_nodes * 9), hidden_size_1, num_layers).to(device)
    rnn_edge = RNN(11, hidden_size_2, num_layers, has_output=True, output_size=7).to(
        device
    )
    rnn_graph.load_state_dict(torch.load(f"models/rnn_graph_model_{max_num_nodes}.pth"))
    rnn_edge.load_state_dict(torch.load(f"models/rnn_edge_model_{max_num_nodes}.pth"))
    with torch.no_grad():
        rnn_graph.eval()
        rnn_edge.eval()
        data = Dataset(max_num_nodes, test=True)
        print(data.data_bfs_adj[graph])
        print(data.data_bfs_edge_dir[graph])
        print(data.data_bfs_offset[graph])
        nodes = data.data_bfs_nodes[graph]
        x_step = torch.ones(batch_size, 1, max_num_nodes * 9)
        y_pred = torch.zeros(batch_size, max_num_nodes, max_num_nodes * 7)
        for i in range(max_num_nodes):
            output_graph = rnn_graph(x_step)
            rnn_edge.hidden = rnn_edge.init_hidden(batch_size)
            rnn_edge.hidden[0, :, :] = output_graph[:, -1, :]
            edge_input_step = torch.ones(batch_size, 11)
            edge_y_pred = torch.zeros(batch_size, max_num_nodes, 9)
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
            y_pred[:, i, :] = edge_y_pred[:, : max_num_nodes * 7]
            x_step = edge_y_pred.unsqueeze(0)

        adj = (
            y_pred[0, :, :max_num_nodes]
            .reshape(max_num_nodes, max_num_nodes)
            .to(torch.int64)
        )
        adj.diagonal().fill_(0)
        adj = adj + adj.T
        edge_dir = y_pred[0, :, max_num_nodes : max_num_nodes * 6]
        edge_dir_splits = torch.split(edge_dir, edge_dir.size(1) // 5, dim=1)
        edge_dir = torch.stack(edge_dir_splits, dim=2)
        edge_dir = torch.argmax(edge_dir, dim=2)

        edge_dir.diagonal().fill_(0)
        mapping = torch.tensor([0, 3, 4, 1, 2])
        edge_dir = edge_dir + mapping[edge_dir].T
        offset = y_pred[0, :, max_num_nodes * 6 :].reshape(max_num_nodes, max_num_nodes)
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


def train(
    model, model_sel, device, learning_rate, epochs, max_num_nodes, training_data
):
    print(f"max nodes: {max_num_nodes}")
    if model_sel == "rnn_mlp":
        rnn = model["nn1"]
        mlp = model["nn2"]
        if not os.path.isfile(
            f"models/rnn_model_{max_num_nodes}.pth"
        ) or not os.path.isfile(f"models/mlp_model_{max_num_nodes}.pth"):
            rnn, mlp, loss_sum = train_rnn_mlp(
                rnn, mlp, device, learning_rate, epochs, max_num_nodes, training_data
            )
    elif model_sel == "rnn_rnn":
        rnn_graph = model["nn1"]
        rnn_edge = model["nn2"]
        if (
            not os.path.isfile(f"models/rnn_graph_model_{max_num_nodes}.pth")
            or not os.path.isfile(f"models/rnn_edge_model_{max_num_nodes}.pth")
            or True
        ):
            rnn_graph, rnn_edge, loss_sum = train_rnn_rnn(
                rnn_graph,
                rnn_edge,
                device,
                learning_rate,
                epochs,
                max_num_nodes,
                training_data,
            )


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model_sel = "rnn_rnn"
    learning_rate = 0.001
    epochs = 1000
    batch_size = 10
    hidden_size_1 = 64
    hidden_size_2 = 64
    num_layers = 4

    for i in range(5):
        test_inference_rnn_rnn(device, hidden_size_1, hidden_size_2, 6, i, num_layers)
    dataset = Dataset(6, test=True)

    max_num_nodes = dataset.max_num_nodes

    data = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model = {}
    if model_sel == "rnn_mlp":
        rnn = RNN(max_num_nodes, 32, 3).to(device)
        mlp = MLP(32, 16, max_num_nodes).to(device)
        model = {"nn1": rnn, "nn2": mlp}
    elif model_sel == "rnn_rnn":
        rnn_graph = RNN((max_num_nodes * 9), hidden_size_1, num_layers).to(device)
        rnn_edge = RNN(
            11, hidden_size_2, num_layers, has_output=True, output_size=7
        ).to(device)
        model = {"nn1": rnn_graph, "nn2": rnn_edge}

    # train(model, model_sel, device, learning_rate, epochs, max_num_nodes, data)

    # test_rnn_rnn(device, data, max_num_nodes, model['nn1'], model['nn2'])

    # test_rnn_mlp(device, data, max_num_nodes, rnn, mlp)
    # test_inference_rnn_mlp(device, rnn, mlp, max_num_nodes, batch_size=batch_size)
