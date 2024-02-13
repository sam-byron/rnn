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
import copy

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

# train_val_split = 0.8
# ordered_en_spa_pairs = copy.deepcopy(en_spa_pairs)
# train_val_pairs = en_spa_pairs[:math.floor(train_val_split*len(en_spa_pairs))]
# test_pairs = en_spa_pairs[math.floor(train_val_split*len(en_spa_pairs))+1:]
# # Shuffling introduces longer sentences which reduces the peformances of the curent
# # model wrt BLEU score but increases accuracy on train and validation set.
# random.shuffle(train_val_pairs)
# val_split = 0.7
# train_pairs = train_val_pairs[:math.floor(val_split*len(train_val_pairs))]
# val_pairs = train_val_pairs[math.floor(val_split*len(train_val_pairs))+1:]

val_split = 0.6
test_split = 0.8
train_pairs = en_spa_pairs[:math.floor(val_split*len(en_spa_pairs))]
val_pairs = en_spa_pairs[math.floor(val_split*len(en_spa_pairs))+1:math.floor(test_split*len(en_spa_pairs))]
test_pairs = en_spa_pairs[math.floor(test_split*len(en_spa_pairs))+1:]

indices = list(range(len(en_spa_pairs)))

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
test_dataset = EnSpaDataset(test_pairs)

# Create dataloader
from torch.utils.data import DataLoader

def encode_transform_batch(source_vocab, target_vocab):

    source_text_transform = lambda src: [source_vocab[token] for token in tokenizer(src)]
    target_text_transform = lambda target: [target_vocab[token] for token in tokenizer(target)]

    def collate_fn(batch):
        src_list, target_list, src_lengths, target_lengths = [], [], [], []
        for _src, _target in batch:
            processed_src = torch.tensor(source_text_transform(_src))
            src_list.append(processed_src)
            src_lengths.append(processed_src.shape[0])
            processed_target = torch.tensor(target_text_transform(_target))
            target_list.append(processed_target)
            # Decrease by 1 bc terminating token will be removed at training time
            target_lengths.append(processed_target.shape[0]-1)
            
        padded_src_list = nn.utils.rnn.pad_sequence(
            src_list, batch_first=True)
        padded_target_list = nn.utils.rnn.pad_sequence(
            target_list, batch_first=True)
        return padded_src_list, padded_target_list, src_lengths
    
    return collate_fn

batch_size = 64
collate_fn = encode_transform_batch(en_vocab, spa_vocab)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
valid_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

# Create model

class RNN(nn.Module):
    def __init__(self, vocab_size, target_vocab_size, embed_dim, target_embed_dim, rnn_hidden_size, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        # It's important to specify to embedding layer the padding index which needs to be
        # ignored when computing gradients. See book "Maching Learning with Pytorch and scikit-learn chapter 15, page 519"
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
        self.target_embedding = nn.Embedding(target_vocab_size, target_embed_dim, padding_idx=0)
        self.rnn_hidden_size = rnn_hidden_size
        self.encoder_rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                           batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(target_embed_dim, 2*rnn_hidden_size, batch_first=True)
        # self.decoder_rnn = nn.LSTM(target_embed_dim, rnn_hidden_size, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(2*rnn_hidden_size, target_vocab_size)
    

    def forward(self, src_lang, target_lang, src_lengths):
        src_embedding = self.embedding(src_lang)
        out = nn.utils.rnn.pack_padded_sequence(src_embedding, src_lengths, enforce_sorted=False, batch_first=True)
        out, (e_hidden, e_cell) = self.encoder_rnn(out)
        target_embedding = self.target_embedding(target_lang)
        # out = nn.utils.rnn.pack_padded_sequence(target_embedding, target_lengths, enforce_sorted=False, batch_first=True)
        e_hidden_cat = torch.cat((e_hidden[-2, :, :], e_hidden[-1, :, :]), dim=1)
        e_cell_cat = torch.cat((e_cell[-2, :, :], e_cell[-1, :, :]), dim=1)
        out, (d_hidden, d_cell) = self.decoder_rnn(target_embedding, (e_hidden_cat.unsqueeze(0), e_cell_cat.unsqueeze(0)))
        # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
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
src_batch, target_batch, src_lengths = next(iter(train_dl))
src_batch_summary = src_batch
target_batch_summary = target_batch
summary(model, input_data=[src_batch_summary, target_batch_summary, src_lengths], verbose=2)

loss_fn = nn.CrossEntropyLoss(reduction='mean')

# Train
from statistics import mean
import gc
from torchtext.data.metrics import bleu_score


def train_procedure(dataloader, model, optimizer, device):
    model.train()
    model.to(device)
    total_acc, total_loss, total_bs = 0, 0, 0
    spa_tokens2str = lambda tokens: [spa_vocab.lookup_token(token) for token in tokens]
    en_tokens2str = lambda tokens: [en_vocab.lookup_token(token) for token in tokens]
    striper = lambda words: [word for word in words if word not in ['baalesh','wa2ef','<pad>']]
    counter = 0
    for src_batch, target_batch, src_lengths in dataloader:
        src_batch = src_batch.to(device)
        target_batch = target_batch.to(device)
        optimizer.zero_grad()
        pred, hidden, cell = model(src_batch, target_batch[:,:-1], src_lengths)
        reordered_pred = pred.permute(0,2,1)
        # Cross-entropy loss function expects inputs to be logits
        loss = loss_fn(reordered_pred, target_batch[:,1:])
        loss.backward()
        optimizer.step()
        total_loss += loss
        candidate_corpus = []
        reference_corpus = []
        max_test_predictions = reordered_pred.argmax(axis=1)
        for trans in max_test_predictions:
            candidate_corpus.append(spa_tokens2str(trans.tolist()))
        for target in target_batch:
            reference_corpus.append([striper(spa_tokens2str(target.tolist()))])
        bs = bleu_score(candidate_corpus, reference_corpus)
        total_bs += bs
        correct = torch.eq(max_test_predictions, target_batch[:,1:])
        acc = torch.mean(torch.sum(correct, 1).float()/correct.shape[1])
        total_acc += acc
        counter += 1

    return total_loss/counter, total_acc/counter, total_bs/counter


def validation_procedure(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        total_acc, total_loss = 0, 0
        counter = 0
        for src_batch, target_batch, src_lengths in dataloader:
            src_batch = src_batch.to(device)
            target_batch = target_batch.to(device)
            pred, hidden, cell = model(src_batch, target_batch[:,:-1], src_lengths)
            reordered_pred = pred.permute(0,2,1)
            loss = loss_fn(reordered_pred, target_batch[:,1:])
            total_loss += loss
            max_test_predictions = reordered_pred.argmax(axis=1)
            correct = torch.eq(max_test_predictions, target_batch[:,1:])
            acc = torch.mean(torch.sum(correct, 1).float()/correct.shape[1])
            total_acc += acc
            counter += 1

    return total_loss/counter, total_acc/counter


TRAIN = True
LOAD = False
SAVE = False
SAMPLE = True
BLEUSCORE = True
PATH = './models/en_spa_translation/en_spa_translation.pt'
if LOAD:
    model = torch.load(PATH)
    if TRAIN:
        model.train()
    else:
        model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
if TRAIN:
    start_time = time.time()
    # device = torch.device("cuda:0")
    for epoch in range(num_epochs):
        loss, acc, bs = train_procedure(train_dl, model, optimizer, device)
        val_loss, val_acc = validation_procedure(valid_dl, model, device)
        if epoch % 1 == 0:
            end_time = time.time()
            print(f'Epoch {epoch} loss: {loss:.3f} acc {acc:0.3f} bs: {bs:.3f}')
            print(f'val_loss: {val_loss:.3f} val_acc: {val_acc:0.3f} time: {(end_time - start_time):.3f} secs')
            start_time = time.time()
            # Save learned model
            if SAVE:
                torch.save(model, PATH)
    

# Evaluate
from torch.distributions.categorical import Categorical
import numpy as np
                
model.eval()


def sample(model, en_sentence, scale_factor=1.0):
    
    model.to('cpu')

    source_text_transform = lambda src: [en_vocab[token] for token in tokenizer(src)]
    en_sentence = torch.tensor(source_text_transform(en_sentence))
    
    target_text_transform = lambda target: [spa_vocab[token] for token in tokenizer(target)]

    spa_sentence = start_translation_token
    spa_sentence_tokens = torch.tensor(target_text_transform(spa_sentence))
    start_trans_token_vec = torch.tensor(target_text_transform(start_translation_token))
    end_trans_token_vec = torch.tensor(target_text_transform(end_translation_token))
    counter = 0
    while spa_sentence_tokens[-1] != end_trans_token_vec and counter < 20:

        translation_logits, h, c = model(en_sentence.unsqueeze(0), spa_sentence_tokens.unsqueeze(0), [len(en_sentence)])
        # translation_logits, h, c = model(en_sentence, spa_sentence_tokens, len(en_sentence))
        # TODO: Sample next token more randomly
        translation_logits = torch.squeeze(translation_logits, 0)
        m = Categorical(logits=translation_logits[-1,:])
        next_word_token = m.sample()
        next_word_token = np.argmax(translation_logits[-1,:].detach().numpy())
        spa_sentence_tokens = torch.cat((spa_sentence_tokens, torch.tensor([next_word_token])))
        counter += 1
    
    spa_seq_len = len(spa_sentence_tokens)
    spa_sentence = ""
    for i in range(spa_seq_len):
        if spa_sentence_tokens[i] != start_trans_token_vec and spa_sentence_tokens[i] !=    end_trans_token_vec:
            next_word = spa_vocab.lookup_token(spa_sentence_tokens[i])
            spa_sentence += next_word + " "

    return spa_sentence

if SAMPLE:
    num_samples = 10
    random_en_spa_pairs = random.sample(en_spa_pairs, num_samples)

    for i in range(num_samples):
        en_sentence = random_en_spa_pairs[i][0]
        spa_translation = sample(model, en_sentence)
        # spa_translation = spa_translation[1:-1]
        print(f'en sentence: {en_sentence} spa translation: {spa_translation}')

    print(sample(model, "How are you?"))
    print(sample(model, "What time is it"))
    print(sample(model, "I want to go home"))
    print(sample(model, "Good night"))

# Offline BLEU Score Evaluation
# Unlike the BLUE score computed during training and evaluation, the offline computation
# takes into account multiple possible translations of the same english sentence thus
# making the score more accurate.
    
if BLEUSCORE:
    # validation_split = 0.2
    # train_pairs = ordered_en_spa_pairs[:math.floor(validation_split*len(ordered_en_spa_pairs))]
    # val_pairs = ordered_en_spa_pairs[math.floor(validation_split*len(ordered_en_spa_pairs))+1:]

    # val_dataset = EnSpaDataset(train_pairs)
    # val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)

    spa_tokens2str = lambda tokens: [spa_vocab.lookup_token(token) for token in tokens]
    en_tokens2str = lambda tokens: [en_vocab.lookup_token(token) for token in tokens]
    striper = lambda words: [word for word in words if word not in ['baalesh','wa2ef','<pad>']]

    model.to(device)
    model.eval()
    with torch.no_grad():
        total_acc, total_loss, total_bs = 0, 0, 0
        counter = 0
        for src_batch, target_batch, src_lengths in valid_dl:
            src_batch = src_batch.to(device)
            target_batch = target_batch.to(device)
            pred, hidden, cell = model(src_batch, target_batch[:,:-1], src_lengths)
            reordered_pred = pred.permute(0,2,1)
            max_test_predictions = reordered_pred.argmax(axis=1)
            candidate_corpus = []
            reference_corpus = []
            last_sentence = []
            ref_idx = -1
            for i in range(batch_size):
                crnt_sentence = src_batch[i].tolist()
                if crnt_sentence != last_sentence:
                    candidate_corpus.append(spa_tokens2str(max_test_predictions[i].tolist()))
                    # Use as a sanity check by including the accurate translation from the reference corpus
                    # in the candidate corpus.
                    # candidate_corpus.append(striper(spa_tokens2str(target_batch[i].tolist())))
                    reference_corpus.append([striper(spa_tokens2str(target_batch[i].tolist()))])
                    ref_idx += 1
                    last_sentence = crnt_sentence
                else:
                    reference_corpus[ref_idx].append(striper(spa_tokens2str(target_batch[i].tolist())))
            bs = bleu_score(candidate_corpus, reference_corpus)
            total_bs += bs
            counter += 1
        bs = total_bs/counter

    print(f'BLEU Score: {bs:.3f}')
