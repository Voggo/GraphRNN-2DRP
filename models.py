import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
from data import DatasetSimple, Dataset


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
    rnn_graph, rnn_edge, device, learning_rate, epochs, max_num_nodes, training_data
):
    optimizer_rnn_graph = torch.optim.Adam(
        list(rnn_graph.parameters()), lr=learning_rate
    )
    optimizer_rnn_edge = torch.optim.Adam(list(rnn_edge.parameters()), lr=learning_rate)
    loss = 0
    rnn_graph.train()
    rnn_edge.train()
    for epoch in range(epochs):
        loss_sum = 0
        for batch in training_data:
            rnn_graph.zero_grad()
            rnn_edge.zero_grad()
            x_bumpy = batch["x"].float().to(device)
            y_bumpy = batch["y"].float().to(device)
            x = torch.flatten(x_bumpy, start_dim=1, end_dim=2).transpose(-2, -1)
            y = torch.flatten(y_bumpy, start_dim=1, end_dim=2).transpose(-2, -1)
            rnn_graph.hidden = rnn_graph.init_hidden(x.size(0))
            output_graph = rnn_graph(x)
            y_pred = torch.zeros(x.size(0), max_num_nodes, max_num_nodes * 3)
            for i in range(max_num_nodes):
                rnn_edge.hidden = rnn_edge.init_hidden(x.size(0))
                rnn_edge.hidden[0, :, :] = output_graph[:, -1, :]
                # burde være udgangspunktet for hidden layer også er inputtet her de tre edge features
                # og de to node features af den næste node. det skal igennem en linear layer og relu før det
                # bliver inputtet til næste hidden layer.
                edge_input = x_bumpy[:, :, 1:, i]
                output_edge = rnn_edge(edge_input.transpose(-2, -1))
                output_edge[:, :, 0:1] = F.sigmoid(output_edge[:, :, 0:1])
                output_edge = output_edge.transpose(-2, -1).flatten(
                    start_dim=1, end_dim=2
                )
                y_pred[:, i, :] = output_edge
            loss_k2 = F.binary_cross_entropy(
                y_pred[:, :, :max_num_nodes], y[:, :, :max_num_nodes]
            )
            loss_l2 = F.mse_loss(y_pred[:, :, max_num_nodes:], y[:, :, max_num_nodes:])
            loss = loss_k2 + loss_l2
            loss_sum += loss.item()
            loss.backward()
            optimizer_rnn_graph.step()
            optimizer_rnn_edge.step()
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, loss: {loss_sum/5}")
    return rnn_graph, rnn_edge, loss_sum


def train(
    model, model_sel, device, learning_rate, epochs, max_num_nodes, training_data
):
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
        rnn_graph, rnn_edge, loss_sum = train_rnn_rnn(
            rnn_graph,
            rnn_edge,
            device,
            learning_rate,
            epochs,
            max_num_nodes,
            training_data,
        )
    print(f"Loss sum: {loss_sum}")


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    model_sel = "rnn_rnn"
    learning_rate = 0.001
    epochs = 1000
    batch_size = 12
    feature_size = 5
    hidden_size_1 = 64
    hidden_size_2 = 64

    data = Dataset(120, 100, 100)
    # print(type(data[0][0]))

    max_num_nodes = data.max_num_nodes

    training_data = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model = {}
    if model_sel == "rnn_mlp":
        rnn = RNN(max_num_nodes, 32, 3).to(device)
        mlp = MLP(32, 16, max_num_nodes).to(device)
        model = {"nn1": rnn, "nn2": mlp}
    elif model_sel == "rnn_rnn":
        rnn_graph = RNN((max_num_nodes * 5) + 5, hidden_size_1, 3).to(device)
        rnn_edge = RNN(5, hidden_size_2, 3, has_output=True, output_size=3).to(device)
        model = {"nn1": rnn_graph, "nn2": rnn_edge}

    train(model, model_sel, device, learning_rate, epochs, max_num_nodes, training_data)

    # test_rnn_mlp(device, training_data, max_num_nodes, rnn, mlp)
    # test_inference_rnn_mlp(device, rnn, mlp, max_num_nodes, batch_size=batch_size)
