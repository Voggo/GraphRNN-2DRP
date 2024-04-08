import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import os
from data import Dataset


class RNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
    ):
        super(RNN, self).__init__()
        self.rnn = torch.nn.GRU(batch_first=True)
        self.out = torch.nn.Linear(100, 1)


    def forward(self, x):
        output, self.hidden = self.rnn(x, self.hidden)
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
                print(loss.item())

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


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    learning_rate = 0.001
    epochs = 1000
    batch_size = 12

    data = Dataset(120, 100, 100)
    # print(type(data[0][0]))

    max_num_nodes = data.max_num_nodes

    training_data = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    rnn = RNN(max_num_nodes, 32, 3).to(device)
    mlp = MLP(32, 16, max_num_nodes).to(device)

    if not os.path.isfile(f"models/rnn_model.pth_{max_num_nodes}") or not os.path.isfile(
        f"models/mlp_model_{max_num_nodes}.pth"
    ):
        rnn, mlp, loss_sum = train_rnn_mlp(
            rnn, mlp, device, learning_rate, epochs, max_num_nodes, training_data
        )

    # test_rnn_mlp(device, training_data, max_num_nodes, rnn, mlp)
    test_inference_rnn_mlp(device, rnn, mlp, max_num_nodes, batch_size=batch_size)
