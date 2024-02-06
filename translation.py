#TODO:
# Strip punctuations from english and spanish

# !wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
# !unzip -q spa-eng.zip

import torch
import torch.nn as nn
from torchinfo import summary
import math
import time
import os
import random

# https://pytorch.org/docs/stable/notes/cuda.html
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

# open file
with open('./spa-eng/spa.txt', 'r') as fp:
    text = fp.read()
    lines = text.split("\n")[:-1]
en_spa_pairs = []
start_translation_token = "baalesh"
end_translation_token = "wa2ef"
for line in lines:
    english, spanish = line.split("\t")
    # Need start and end token for model to know when to begin and end translation
    spanish = start_translation_token + " " + spanish + " " + end_translation_token
    en_spa_pairs.append((english, spanish))

# For quick testing
# en_spa_pairs = en_spa_pairs[0:10000]   

validation_split = 0.8
random.shuffle(en_spa_pairs)
train_pairs = en_spa_pairs[:math.floor(validation_split*len(en_spa_pairs))]
val_pairs = en_spa_pairs[math.floor(validation_split*len(en_spa_pairs))+1:]

# Tokenize
from text_preprocessing import tokenizer, vocab_builder
from collections import Counter
from torch.utils.data import Dataset

en_token_counts = Counter()
spa_token_counts = Counter()
for (en, spa) in en_spa_pairs:
    en_tokens = tokenizer(en)
    en_token_counts.update(en_tokens)
    spa_tokens = tokenizer(spa)
    spa_token_counts.update(spa_tokens)

print('Enlish vocab size', len(en_token_counts))
print('Spanish vocab size', len(spa_token_counts))

en_vocab = vocab_builder(en_token_counts)
spa_vocab = vocab_builder(spa_token_counts)

# Create dataset
class EnSpaDataset(Dataset):
    def __init__(self, eng_spa_pairs):
        self.eng_spa_pairs = eng_spa_pairs

    def __len__(self):
        return len(self.eng_spa_pairs)
    
    def __getitem__(self, idx):
        en, spa = self.eng_spa_pairs[idx]
        return en, spa
    
train_dataset = EnSpaDataset(train_pairs)
val_dataset = EnSpaDataset(val_pairs)

# Create dataloader
from torch.utils.data import DataLoader

def encode_transform_batch(source_vocab, target_vocab):

    source_text_transform = lambda src: [source_vocab[token] for token in tokenizer(src)]
    target_text_transform = lambda target: [target_vocab[token] for token in tokenizer(target)]

    def collate_fn(batch):
        src_list, target_list = [], []
        for _src, _target in batch:
            src_list.append(torch.tensor(source_text_transform(_src)))
            target_list.append(torch.tensor(target_text_transform(_target)))
        padded_src_list = nn.utils.rnn.pad_sequence(
            src_list, batch_first=True)
        padded_target_list = nn.utils.rnn.pad_sequence(
            target_list, batch_first=True)
        return padded_src_list, padded_target_list
    
    return collate_fn

batch_size = 64
collate_fn = encode_transform_batch(en_vocab, spa_vocab)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Create model

class RNN(nn.Module):
    def __init__(self, vocab_size, target_vocab_size, embed_dim, target_embed_dim, rnn_hidden_size, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.target_embedding = nn.Embedding(target_vocab_size, target_embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.encoder_rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                           batch_first=True)
        self.decoder_rnn = nn.LSTM(target_embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, target_vocab_size)
    

    def forward(self, src_lang, target_lang):
        src_embedding = self.embedding(src_lang)
        out, (e_hidden, e_cell) = self.encoder_rnn(src_embedding)
        target_embedding = self.target_embedding(target_lang)
        out, (d_hidden, d_cell) = self.decoder_rnn(target_embedding, (e_hidden, e_cell))
        out = nn.Dropout(p=0.5)(out)
        out = self.fc(out)

        return out, d_hidden, d_cell
    

    def init_hidden(self, batch_size):
        # 1 represents number of hidden layers in rnn
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        hidden = torch.squeeze(hidden, 1)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.squeeze(cell, 1)
        return hidden, cell

# embed_dim = math.floor(len(en_vocab)*0.05) 
embed_dim = 256
# target_embed_dim = math.floor(len(spa_vocab)*0.05)
target_embed_dim = 256
# rnn_hidden_size = 512
rnn_hidden_size = 1024
device = torch.device("cuda:0")
model = RNN(len(en_vocab), len(spa_vocab), embed_dim, target_embed_dim, rnn_hidden_size, batch_size, device)
print(model)
src_batch, target_batch = next(iter(train_dl))
src_batch_summary = src_batch
target_batch_summary = target_batch
summary(model, input_data=[src_batch_summary, target_batch_summary], verbose=2)

loss_fn = nn.CrossEntropyLoss(reduction='mean')

# Train
from statistics import mean
import gc
num_epochs = 15

def train_procedure(dataloader, model, optimizer, device):
    model.train()
    model.to(device)
    total_acc, total_loss = 0, 0
    counter = 0
    for src_batch, target_batch in dataloader:
        src_batch = src_batch.to(device)
        target_batch = target_batch.to(device)
        optimizer.zero_grad()
        pred, hidden, cell = model(src_batch, target_batch[:,:-1])
        reordered_pred = pred.permute(0,2,1)
        # Cross-entropy loss function expects inputs to be logits
        loss = loss_fn(reordered_pred, target_batch[:,1:])
        loss.backward()
        optimizer.step()
        total_loss += loss

        max_test_predictions = reordered_pred.argmax(axis=1)
        mislabeled = torch.ne(max_test_predictions, target_batch[:,1:])
        acc = 0
        crnt_batch_size = len(mislabeled[:,0])
        batch_acc = []
        for i in range(crnt_batch_size):
            acc = [1 if x == False else 0 for x in mislabeled[i,:]]
            acc = mean(acc)
            batch_acc.append(acc)
        total_acc += mean(batch_acc)

        counter += 1
        del src_batch
        del target_batch
        del pred
        del hidden
        del cell
        del reordered_pred
    return total_loss/counter, total_acc/counter


def validation_procedure(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        total_acc, total_loss = 0, 0
        counter = 0
        for src_batch, target_batch in dataloader:
            src_batch = src_batch.to(device)
            target_batch = target_batch.to(device)
            pred, hidden, cell = model(src_batch, target_batch[:,:-1])
            reordered_pred = pred.permute(0,2,1)
            loss = loss_fn(reordered_pred, target_batch[:,1:])
            total_loss += loss
            max_test_predictions = reordered_pred.argmax(axis=1)
            mislabeled = torch.ne(max_test_predictions, target_batch[:,1:])
            acc = 0
            crnt_batch_size = len(mislabeled[:,0])
            batch_acc = []
            for i in range(crnt_batch_size):
                acc = [1 if x == False else 0 for x in mislabeled[i,:]]
                acc = mean(acc)
                batch_acc.append(acc)
            total_acc += mean(batch_acc)
            counter += 1
            del src_batch
            del target_batch
            del pred
            del hidden
            del cell
            del reordered_pred
    return total_loss/counter, total_acc/counter


TRAIN = True
LOAD = False
SAVE = False
PATH = './models/en_spa_translation/en_spa_translation.pt'
if LOAD:
    model = torch.load(PATH)
    if TRAIN:
        model.train()
    else:
        model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if TRAIN:
    start_time = time.time()
    # device = torch.device("cuda:0")
    for epoch in range(num_epochs):
        loss, acc = train_procedure(train_dl, model, optimizer, device)
        gc.collect()
        torch.cuda.empty_cache()
        val_loss, val_acc = validation_procedure(valid_dl, model, device)
        if epoch % 1 == 0:
            end_time = time.time()
            print(f'Epoch {epoch} loss: {loss:.4f} acc {acc:0.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:0.4f} time: {(end_time - start_time):.3f} secs')
            start_time = time.time()
            # Save learned model
            if SAVE:
                torch.save(model, PATH)
    

# Evaluate
from torch.distributions.categorical import Categorical
                
model.eval()


def sample(model, en_sentence, scale_factor=1.0):
    
    model.to('cpu')

    source_text_transform = lambda src: [en_vocab[token] for token in tokenizer(src)]
    en_sentence = torch.tensor(source_text_transform(en_sentence))

    # next_word = start_translation_token
    target_text_transform = lambda target: [spa_vocab[token] for token in tokenizer(target)]
    # next_word = torch.tensor(target_text_transform(next_word))

    seq_length = 10
    spa_sentence = start_translation_token + " "
    spa_sentence_tokens = torch.tensor(target_text_transform(spa_sentence))
    hidden, cell = model.init_hidden(1)
    end_trans_token_vec = torch.tensor(target_text_transform(end_translation_token))
    counter = 0
    while spa_sentence_tokens[-1] != end_trans_token_vec and counter < 10:

        translation_logits, h, c = model(en_sentence, spa_sentence_tokens, hidden, cell)
        # translation_logits = torch.nn.functional.softmax(translation_logits, dim=1)
        m = Categorical(logits=translation_logits)
        next_word_token = m.sample()
        next_word = spa_vocab.lookup_token(next_word_token[-1])
        spa_sentence += next_word + " "
        spa_sentence_tokens = torch.cat((spa_sentence_tokens, next_word_token))
        # next_word = torch.tensor(target_text_transform(next_word))
        counter += 1
    
    return spa_sentence

print(sample(model, "How are you?"))
print(sample(model, "What time is it"))
print(sample(model, "I want to go home"))
print(sample(model, "Good night"))