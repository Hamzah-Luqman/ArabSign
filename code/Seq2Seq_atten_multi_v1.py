import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F

import os
import random
import codecs
import nltk

import pandas as pd

from utils import save_checkpoint, load_checkpoint

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from generator import CustomCSVDataset
from torch.utils.data import DataLoader, Dataset
from torchmetrics import WordErrorRate  as WER


def FindMaxLength(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
    return maxLength

def readCaptions(filepath):
    # Add all the captions.
    captions = []
    f = open(filepath, "r", encoding='utf-8')
    for line in f.readlines():
        lineWords = line.split()
        lineWords.insert(0, '<SOS>')
        lineWords.insert(len(lineWords), '<EOS>')
        captions.append(lineWords)


    sentenceMaxLengh = FindMaxLength(captions)
    print(sentenceMaxLengh)    
    
    # padding
    paddedCaptions = []
    for caption in captions:
        if len(caption) < sentenceMaxLengh:
            caption = caption + ['<PAD>'] * (sentenceMaxLengh - len(caption))
        paddedCaptions.append(caption)
    #print(paddedCaptions)
    captions = paddedCaptions
    
    vocab = set() # Total unique words including <SOS>, <EOS>, <PAD> forms the vocab. 
    for caption in captions:
        #print(caption, f'len = {len(caption)}')
        for token in caption:
            vocab.add(token)
    print(f'\nVocab:\n{vocab} len = {len(vocab)}')

    # Mapping string/word to an index.
    captionToIndex = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    }

    temp = {}
    idx = 3 # Since indices 0,1,2 are already reversed for tokens <PAD>, <SOS>, <EOS> respectively.
    for caption in captions:
        for tok in caption:
            if tok not in ['<PAD>', '<SOS>', '<EOS>'] and tok not in temp:
                temp[tok] = idx
                idx += 1
                
    captionToIndex.update(temp)
    #print(f'\nString-to-index mapping:\n{captionToIndex}\n')

    # Mapping index to string/word.
    indexToCaption = {value : key for (key, value) in captionToIndex.items()}
    #print(f'\nIndex-to-string mapping:\n{indexToCaption}\n')

    return captions, captionToIndex, indexToCaption, sentenceMaxLengh

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ######self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, bidirectional=True, dropout = p)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size
        x = x.permute(1, 0,2)
        embedding = x #self.dropout(x)

        encoder_states, hidden = self.rnn(embedding)
        encoder_states = (encoder_states[:, :, :self.hidden_size] +
                   encoder_states[:, :, self.hidden_size:])
        cell = ''
        return encoder_states, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(hidden_size  + embedding_size, hidden_size, num_layers, dropout = p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attention = Attention(hidden_size)
        self.energy = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, input, encoder_outputs, last_hidden, cell):
        #x = x.unsqueeze(0)
        ##print('x: ', x.shape)
        # x: (1, N) where N is the batch size
        #embedding = self.embedding(x)
        #embedding = self.dropout(self.embedding(x))
        embedded = self.embedding(input).unsqueeze(0)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, num_layers):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_layers = num_layers
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = 150 #len(english.vocab)
        device ="cuda:0"
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        guesses= torch.zeros(target_len, batch_size).to(device)
        ##print('outputs size:', outputs.shape)
        encoder_output, hidden, cell = self.encoder(source)
        hidden = hidden[:self.num_layers]
        #print('hidden main: ', hidden.shape)
        # First input will be <SOS> token
        x = target[0]
         
        ##print('seq2seq target[0]: ', x.shape)
        for t in range(1, target_len):
            #print(x)
            # At every time step use encoder_states and update hidden, cell
            #print(hidden)
            output, hidden, attn_weights = self.decoder(x, encoder_output, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)
            guesses[t] = best_guess
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs, guesses

 
    


def saveNumpyToCSV(data, filePath):
    pd.DataFrame(data).to_csv(filePath, sep=',')

def createFolder(folderPath):
    if os.path.exists(folderPath) ==  False:
        os.makedirs(folderPath)
def savePredictionsToTxt(data, filePath):
    with codecs.open(filePath, 'w', encoding='utf-8') as outfile:
        for x, y in data:
            outfile.write(x + "\t" + y +"\n")
    

def remove_special_tokens(sentence):
    cleaned_sent = ''
    tokenized_sent = sentence.split()
    for tok in tokenized_sent:
        if tok not in ['<PAD>', '<SOS>', '<EOS>']:
            cleaned_sent += tok + ' '
    return cleaned_sent


def decode_sentence(sentence, indexToCaption):
    sent = ''
    #print('Sent: ', sentence)
    for idx in sentence:
        #print(idx.argmax(0).item())
        if isinstance(idx, torch.Tensor):
            #print('indx torch:', int(idx.item()))
            sent += indexToCaption[int(idx.item())] + ' '
        else:
            #print(indexToCaption[idx])
            sent += indexToCaption[idx] + ' '
    
    #print(sent)
    sent = remove_special_tokens(sent)
    #print(sent)
    return sent

def eval_getCaption(model, n_layers, source, target, device,  captionToIndex):
    target_vocab_size = 150
    # First input will be <SOS> token
    batch_size = target.shape[1]
    seq_len = target.shape[0]
    # print('seq2seq seq_len: ', seq_len)
    # print('img :', source.shape)

    guesses= torch.zeros(seq_len, batch_size).to(device)
    outputs = torch.zeros(seq_len, batch_size, target_vocab_size).to(device)

    encoder_states, hidden, cell = model.encoder(source)
    hidden = hidden[:n_layers]
    #print('encoder_states :', encoder_states.shape)
    #predicted_caption = target[0]#[captionToIndex['<SOS>']]
    x =target[0]
    #print('target: ',target[0] )
    for t in range(1, seq_len):
        #x = torch.LongTensor([predicted_caption[-1]]).to(device)
        #print(x)
        output, hidden, cell = model.decoder(x, encoder_states, hidden, cell)
        outputs[t] = output
        # Get the best word the Decoder predicted (index in the vocabulary)
        best_guess = output.argmax(1)
        guesses[t] = best_guess
        #predicted_caption.append(best_guess)
        x = best_guess
        #print('best_guess: ', guesses[t] )
        #print(best_guess)
    return outputs, guesses

def eval_model_2(test_loader, model, n_layers, criterion, device, captionToIndex, indexToCaption, modelName, printPredictions = False):
    wer = WER()
    model.eval()
    all_predicted = []
    all_true = []
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, (img_frames, captions) in enumerate(test_loader):
            inp_data = img_frames.to(device)
            target = captions.to(device).permute(1,0)
            output, predictions = eval_getCaption(model, n_layers, inp_data, target, device, captionToIndex)
            # output, predictions = model.encoder(inp_data, target)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output, target)
            # print(output.shape)
            # print(target.shape)
            predictions = predictions.permute(1,0)[:,1:]
            # print(predictions.shape)
            target = target.reshape(img_frames.shape[0],-1)
            # print(target.shape)
            caption_2 = captions[:, 1:]
            # print(caption_2.shape)
        
            for cap, pred in zip(caption_2, predictions): 
                true_caption = decode_sentence(cap.to('cpu').numpy(), indexToCaption)
                predicted_caption = decode_sentence(pred.to('cpu').numpy(), indexToCaption)
                # print(f'True caption: {true_caption}')
                # print(f'Predicted caption: {predicted_caption}')
                all_true.append(true_caption)
                all_predicted.append(predicted_caption)
                count += 1
            #print(count)    


            total_loss += loss
        
    
    if printPredictions:
        for true_c, pred_c in zip(all_true, all_predicted):
            print(f'ACTUAL:    {true_c}')
            print(f'PREDICTED: {pred_c}')
            print()
    
    error_rate = wer(all_predicted, all_true).item()
    total_loss = total_loss / len(test_loader)
    return error_rate, all_true, all_predicted, total_loss



def startTraining(trainPath, noInputFeatures, expPath, modelName, save_model):
    device = "cuda:0"

    # Training hyperparameters
    num_epochs = 150
    learning_rate = 1e-3
    batch_size = 16

    # Model hyperparameters
    input_size_encoder = 80 #len(german.vocab)

    output_size = 150 #len(english.vocab)
    encoder_embedding_size = noInputFeatures
    decoder_embedding_size = 300
    hidden_size = 512
    num_layers = 3
    enc_dropout = 0.4
    dec_dropout = 0.4


    # Read caption/ground truth
    captions, captionToIndex, indexToCaption, sentenceMaxLengh = readCaptions(groundTruth)
    input_size_decoder =150 #len(english.vocab)
    # Load data
    trainDataset = CustomCSVDataset(trainPath, captionToIndex, captions, None,'train', 0.2, True)    
    train_iterator = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    print('Train data size:', next(iter(train_iterator))[0].shape)

    valDataset = CustomCSVDataset(trainPath, captionToIndex, captions, None, 'val', 0.2, True)    
    valid_iterator = DataLoader(valDataset, batch_size=batch_size, shuffle=True)   
    print('val data size:', next(iter(valid_iterator))[0].shape)

    ####
    writer = SummaryWriter(f"runs/loss_plot")
    step = 0

    encoder_net = Encoder(
        input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
    ).to(device)

    decoder_net = Decoder(
        input_size_decoder,
        decoder_embedding_size,
        hidden_size,
        output_size,
        num_layers,
        dec_dropout,
    ).to(device)

    model = Seq2Seq(encoder_net, decoder_net, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = captionToIndex["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # if load_model:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


    train_losses = []
    train_WER = []
    val_WER = []
    best_val_loss = 100 #random initial value
    for epoch in range(num_epochs):
        running_loss = 0.0
        trainWER = 0.0

        model.train()
        prog_bar = tqdm(enumerate(train_iterator), total=len(train_iterator), leave=True)
        for batch_idx, (img_frames, cap) in prog_bar:
            # Get input and targets and get to cuda
            inp_data = img_frames.to(device)
            target = cap.to(device).permute(1,0)
            ##print('inp_data size: ', inp_data.shape)
            ##print('Target size: ', target.shape)
            # Forward prop
            output, _ = model(inp_data, target)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            running_loss += loss.item()

            # Back prop
            loss.backward()


            # Gradient descent step
            optimizer.step()

            # Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1
            prog_bar.set_description(f'Epoch {epoch}/{num_epochs}')
        
        tr_loss = running_loss / len(train_iterator)
        train_losses.append(tr_loss)
        print(f'\ttrain_loss = {tr_loss:.6f}')
        trainWER = 0
        train_WER.append(trainWER)

        # validation WER  
        if valid_iterator != None:
            device = "cuda:0"
            valWER, _, _, val_loss = eval_model_2(valid_iterator, model.to(device), num_layers, criterion, device, captionToIndex, indexToCaption, modelName)
            print(f'\t Val loss = {val_loss}, Val WER = {valWER:.3f}')
            val_WER.append([val_loss, valWER])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_model == True:
                    torch.save(model, os.path.join(expPath, "model.pt"))

        else:
            val_WER.append(0)

    train_metrics = np.transpose([train_losses, train_WER, val_WER])
    #train_metrics =np.transpose([train_losses])
    saveNumpyToCSV(train_metrics, os.path.join(expPath, "Metrics.csv"))

    return model, num_layers, captions, captionToIndex, indexToCaption, criterion



def blue(candidates, reference):
    score_1 = 0.0
    score_2 = 0.0
    score_3 = 0.0
    score_4 = 0.0
    n = len(candidates)
    for cand, ref in zip(candidates, reference):
        cand = cand.split(' ')
        cand = ' '.join(cand).split()
        # c = len(cand)
        #print(cand)
        ref = ref.split(' ')
        ref = ' '.join(ref).split()
        #print(ref)
        # r = len(ref)
        score_1 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(1, 0, 0, 0))
        score_2 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 1, 0, 0))
        score_3 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 0, 1, 0))
        score_4 += nltk.translate.bleu_score.sentence_bleu([ref], cand, weights=(0, 0, 0, 1))
        # print(f"Bleu-1 score {score_1 :.2f}")
        # print(f"Bleu-2 score {score_2 :.2f}")
        # print(f"Bleu-3 score {score_3 :.2f}")
        # print(f"Bleu-4 score {score_4 :.2f}")

    return score_1/n, score_2/n, score_3/n,score_4/n




 

###

# from GPUtil import showUtilization as gpu_usage
# gpu_usage() 
# torch.cuda.empty_cache()

# from numba import cuda
# cuda.select_device(0)
# cuda.close()
# cuda.select_device(0)



def startExperiment(expPath, trainPath, testPath, groundTruth, nFramesNorm, diFeature, modelName, load_model, save_model):
 
    model, num_layers, captions, captionToIndex, indexToCaption, criterion = startTraining(trainPath, diFeature, expPath, expPath, save_model)

    testDataset = CustomCSVDataset(testPath, captionToIndex, captions)
    test_iterator = DataLoader(testDataset, batch_size=16, shuffle=False)
    print('test data size:', next(iter(test_iterator))[0].shape)

    #torch.cuda.empty_cache()
    if load_model == True:
        model = torch.load(os.path.join(expPath, "model.pt"))


    device ="cuda:0"
    wordErrorRate, all_true, all_predicted, test_loss = eval_model_2(test_iterator, model, num_layers, criterion, device, captionToIndex, indexToCaption, modelName)
    pred_Truth = np.transpose([all_true, all_predicted])
    #print(pred_Truth)
    savePredictionsToTxt(pred_Truth, os.path.join(expPath, "prediction.txt"))
    print(f'Test Loss = {test_loss}, Test WER = {wordErrorRate}') 
    f= open(os.path.join(expPath, 'testWER.txt'),"a")
    #f.write(' '.join(expInfo)+'\n')
    f.write('WER:\t' +str(wordErrorRate)  + '\n')



    
    score_1, score_2, score_3,score_4 = blue(all_predicted, all_true)
    print(f"Bleu-1 score {score_1 :.2f}")
    print(f"Bleu-2 score {score_2 :.2f}")
    print(f"Bleu-3 score {score_3 :.2f}")
    print(f"Bleu-4 score {score_4 :.2f}")

    f.write(f"Bleu-1 score {score_1 :.2f}"+ '\n')
    f.write(f"Bleu-2 score {score_2 :.2f}"+ '\n')
    f.write(f"Bleu-3 score {score_3 :.2f}"+ '\n')
    f.write(f"Bleu-4 score {score_4 :.2f}"+ '\n')

    f.close()

 ################




diVideoSet = {"dataset" : "ArabSign",
    "modelName": "EncoderDecoderAttention", #EncoderDecoderAttention EncoderDecoder
    "nClasses" : 50,   # number of classes
    "nFramesNorm" : 80,    # number of frames per video
    "nMinDim" : 224,   # smaller dimension of saved video-frames
    "tuShape" : (224, 224), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0,# seconds 
    "reshape_input": False}  #True: if the raw input is different from the requested shape for the model
save_model = True
load_model = True

    # feature extractor. 
#diFeature = {"sName" : "vgg16",
 #    "tuInputShape" : (224, 224, 3),
  #   "tuOutputShape" : (4096, )} 

diFeature = {"sName" : "mobilenet",
    "tuInputShape" : (224, 224, 3),
    "tuOutputShape" : (1024, )} 

#diFeature = {"sName" : "inception",
#    "tuInputShape" : (224, 224, 3),
#    "tuOutputShape" : (2048, )} 

#path of the extracted features 
dataSetHomePath = '/home/g202002320/Datasets/ArSL-Continuous/80/features/images/mobilenet/color'
groundTruth = './data/groundTruth.txt' # text file with the unique sentences


# signer dependant all
#trainPath_all = [dataSetHomePath+'/all_train.csv']
#testPath_all = [dataSetHomePath+'/all_test.csv']


# Signer independent testing 
trainPath_all = [dataSetHomePath+'/train_01.csv', dataSetHomePath+'/train_02.csv', dataSetHomePath+'/train_03.csv', dataSetHomePath+'/train_04.csv', dataSetHomePath+'/train_05.csv', dataSetHomePath+'/train_06.csv']
testPath_all = [dataSetHomePath+'/01.csv',dataSetHomePath+'/02.csv' , dataSetHomePath+'/03.csv', dataSetHomePath+'/04.csv', dataSetHomePath+'/05.csv', dataSetHomePath+'/06.csv']

# Uncomment for Signer dependent testing 
#trainPath_all = [dataSetHomePath+'/01_train.csv', dataSetHomePath+'/02_train.csv', dataSetHomePath+'/03_train.csv', dataSetHomePath+'/04_train.csv', dataSetHomePath+'/05_train.csv', dataSetHomePath+'/06_train.csv', dataSetHomePath+'/all_train.csv']
#testPath_all = [dataSetHomePath+'/01_test.csv', dataSetHomePath+'/02_test.csv', dataSetHomePath+'/03_test.csv', dataSetHomePath+'/04_test.csv', dataSetHomePath+'/05_test.csv', dataSetHomePath+'/06_test.csv', dataSetHomePath+'/all_test.csv']

# Name of folders to save results in
expPath_all = ['ArabSign_encDecAtten_SI_test1','ArabSign_encDecAtten_SI_test2','ArabSign_encDecAtten_SI_test3','ArabSign_encDecAtten_SI_test4','ArabSign_encDecAtten_SI_test5', 'ArabSign_encDecAtten_SI_test6']
expPath_all = ['ArabSign_encDecAtten_SI_test1, ArabSign_encDecAtten_SI_test2', "ArabSign_encDecAtten_SI_test3", "ArabSign_encDecAtten_SI_test4", "ArabSign_encDecAtten_SI_test5", "ArabSign_encDecAtten_SI_test6"]

i = 0
while i < len(expPath_all):
    print(expPath_all[i])
    expPath = expPath_all[i]
    trainPath = trainPath_all[i]
    testPath = testPath_all[i]
    expPath = os.path.join(os.getcwd(),'results',expPath)
    createFolder(expPath)
    startExperiment(expPath, trainPath, testPath, groundTruth, diVideoSet['nFramesNorm'] , diFeature['tuOutputShape'][0], diVideoSet['modelName'], load_model, save_model)
    i = i + 1
  