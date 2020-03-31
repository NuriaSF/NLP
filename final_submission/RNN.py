import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, vocab_dim, text, hidden_dim=150, output_dim=2, num_layers=3, dropout=0.2):
        super().__init__()

        self.vocab_dim = vocab_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        embedding_dim = 100
        vocab = text.vocab
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        # num_layers: num of RNN; hidden_size: num of nodes
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                           num_layers=3, dropout=dropout, bidirectional=True)
        self.hidden_dim = hidden_dim
        # [5: feature dim; 2: bidirectional]
        self.input_dim = 5 * 2 * self.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, int(self.input_dim/2)),
            nn.Linear(int(self.input_dim/2), output_dim))

    def forward(self, question1, question2):

        # question = [sent len, batch size]
        embedded1 = self.embedding(question1)
        embedded2 = self.embedding(question2)

        # embedded = [sent len, batch size, emb dim]
        output1, _ = self.rnn(embedded1)
        output2, _ = self.rnn(embedded2)

        # output = [sent len, batch size, hid dim]
        features = torch.cat((output1[-1:, :, :],
                              torch.abs(output1[-1:, :, :] -
                                        output2[-1:, :, :]),
                              output2[-1:, :, :],
                              output1[-1:, :, :]*output2[-1:, :, :],
                              (output1[-1:, :, :]+output2[-1:, :, :])/2), 2)

        # output = [batch size, 5 * 2 * hidden dim]
        y = self.classifier(features)
        return y.squeeze(0)
