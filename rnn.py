import torch
import torch.nn as nn


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, 
                          hidden_size, 
                          num_layers=2, 
                          batch_first=True)
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1, :]
        out = self.fc(out)
        return out

class embedRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, bidirectional=False, rnn_type="lstm", num_layers=1):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim, 
                                      padding_idx=0) 
        if rnn_type == "simple":
            self.rnn = nn.RNN(embed_dim, 
                            rnn_hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                            batch_first=True,
                            num_layers=num_layers,bidirectional=bidirectional)
        elif rnn_type == "gru":
            self.gru = nn.GRU(embed_dim, rnn_hidden_size, num_layers=num_layers, batch_first=True)
        if bidirectional:
            self.fc1 = nn.Linear(rnn_hidden_size*2, fc_hidden_size)
        else:
            self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden, cell) = self.rnn(out)
        if self.bidirectional:
            out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            out = hidden[-1, :, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out