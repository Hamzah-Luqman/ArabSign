from unicodedata import bidirectional
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




class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout) -> None:
        super().__init__()
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional= True, dropout=dropout)       
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        encoder_states, hidden = self.lstm(x)
        encoder_states = (encoder_states[:, :, :self.hidden_size] +
            encoder_states[:, :, self.hidden_size:])


        return encoder_states, hidden


class DecoderRNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_size, hidden_size,  num_layers, dropout, vocab_size) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional= True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size*2, out_features=vocab_size)
        
        
    def forward(self, captions, enc_hidden):

        captions = captions.unsqueeze(0) 
        
        embeddings = self.embed(captions)

        outputs, hidden  = self.lstm(embeddings, enc_hidden) #
        cell = ''
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden 


