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
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=7):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)       


    def forward(self, x):
        
        outputs, hidden = self.gru(x)

        return outputs, hidden 


class DecoderRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size, num_layers, dropout, vocab_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, captions, enc_hidden, enc_cell):

        captions = captions.unsqueeze(0) 
        
        embeddings = self.embed(captions)
        embeddings = F.relu(embeddings)

        outputs, hidden = self.gru(embeddings, enc_hidden) 
        
        predictions = self.softmax(self.out(outputs[0]))

        return predictions, hidden

