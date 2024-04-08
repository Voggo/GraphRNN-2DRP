import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from data import Dataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Define a collate function to pad your batches
def collate_fn(batch):
    # Sort the batch in the descending order

    sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    # Separate sequences and labels
    sequences, y = zip(*sorted_batch)
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, y


data = Dataset(12, 100, 100)
print(type(data[0][0]))

training_data = torch.utils.data.DataLoader(
    data, batch_size=4, shuffle=False, collate_fn=collate_fn
)

for batch, y in training_data:
    print(batch, y)

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,):
        super(RNN, self).__init__()
        self.rnn = torch.nn.GRU(, batch_first=True)
        self.out = torch.nn.Linear(100, 1)


    def forward(self, x):
        output, _ = self.rnn(x)
        return output
