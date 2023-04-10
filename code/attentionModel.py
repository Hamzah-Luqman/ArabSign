import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from torchmetrics import WordErrorRate  as WER

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random






class Attention(nn.Module):
    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = torch.nn.Linear(self.hidden_size, self.hidden_size)
    

    def forward(self, hidden, enc_outputs):
        energy = self.attention(enc_outputs)
        #print('energy: ', energy.shape)

        attention_energy = torch.sum(hidden * energy, dim=2)
        #print('attention_energy: ', attention_energy.shape)
        attention_energy = attention_energy.t() # Transposing the attention_energy tensor.
        softmax_scores = nn.Softmax(dim=1)(attention_energy).unsqueeze(1)
        #print('softmax_scores: ', softmax_scores.shape)
        return softmax_scores


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)       


    def forward(self, x):
        # Shape of x: (frames, batch_size, input_size) i.e. (80, batch_size, 4096)
        
        outputs, (hidden, cell) = self.lstm(x)
        # print('================= Encoder ==============')
        # print(outputs.shape)
        # print(hidden.shape)
        # print(cell.shape)
        # Shape of outputs: (80, batch_size, 1024)
        # Shape of hidden: (3, batch_size, 1024)
        # Shape of cell: (3, batch_size, 1024)

        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, dropout, vocab_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        
        self.attention_model = Attention(hidden_size)
        

    def forward(self, captions, enc_hidden, enc_cell):
        # We will give decoder one word at a time => seq_len = 1

        # Shape of captions: (batch_size,). But we need to create a sequence. Hence, we will shape it into (1, batch_size)
        # Shape of enc_hidden: (2, batch_size, 512)
        # Shape of enc_cell: (2, batch_size, 512)

        captions = captions.unsqueeze(0) # Shaping into (1, batch_size)
        print('captions: ', captions.shape)
        embeddings = self.embed(captions)
        print('embeddings : ', embeddings.size)
        # Shape of embeddings: (seq_len, batch_size, embedding_dim)

        outputs, (hidden, cell) = self.lstm(embeddings, (enc_hidden, enc_cell)) # We pass the enc_hidden & enc_cell to the hidden states of our decoder as initial states
        # Shape of outputs: (1, batch_size, hidden_size) i.e. (1, batch_size, 512)
        # Shape of hidden: (num_layers, batch_size, hidden_size) i.e. (2, batch_size, 512)
        # Shape of cell: (num_layers, batch_size, hidden_size) i.e. (2, batch_size, 512)

        attention_weights = self.attention_model(outputs, enc_hidden)
        context = attention_weights.bmm(enc_hidden.transpose(0,1))
        outputs = outputs.squeeze(0)
        context = context.squeeze(1)
        concat_input =  torch.cat((outputs, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        predictions = self.fc(concat_output)
        predictions = predictions.squeeze(0)
        # Shape of predictions: (vocab_size) i.e. (45,)
        # print('================= Decoder ==============')
        # print('enc_hidden: ', enc_hidden.shape)
        # print(embeddings.shape)
        # print(outputs.shape)
        # print(hidden.shape)
        # print(cell.shape)
        # print(attention_weights.shape)
        # print('concat_input: ', concat_input.shape)
        return predictions, hidden, cell

