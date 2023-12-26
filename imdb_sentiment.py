# coding: utf-8


import sys
from python_environment_check import check_packages
import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data.dataset import random_split
from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
# from torchdata.dataloader2 import DataLoader2
import time
from torchinfo import summary
from rnn import RNN, embedRNN
from text_preprocessing import tokenizer, vocab_builder, encode_transform_batch

# ## Package version checks

# Check recommended package versions

d = {
    'torch': '1.8.0',
    'torchtext': '0.10.0'
}
check_packages(d)


# # Chapter 15: Modeling Sequential Data Using Recurrent Neural Networks (Part 2/3)

# **Outline**
# 
#   - [Project one -- predicting the sentiment of IMDb movie reviews]
#     - [Preparing the movie review data](#Preparing-the-movie-review-data)
#     - [Embedding layers for sentence encoding](#Embedding-layers-for-sentence-encoding)
#     - [Building an RNN model](#Building-an-RNN-model)
#     - [Building an RNN model for the sentiment analysis task](#Building-an-RNN-model-for-the-sentiment-analysis-task)
#       - [More on the bidirectional RNN](#More-on-the-bidirectional-RNN)




# Step 1: load and create the datasets

train_dataset = IMDB(split='train')
test_dataset = IMDB(split='test')
# https://pytorch.org/text/stable/datasets.html#imdb

torch.manual_seed(1)
train_dataset, valid_dataset = random_split(
    list(train_dataset), [20000, 5000])

train_size = 20000
valid_size = 5000
test_size = 25000


## Step 2: find unique tokens (words)

token_counts = Counter()

for label, line in train_dataset:
    tokens = tokenizer(line)
    token_counts.update(tokens)
 
    
print('Vocab-size:', len(token_counts))


## Step 3: encoding each unique token into integers

vocabulary = vocab_builder(token_counts)

print([vocabulary[token] for token in ['this', 'is', 'an', 'example']])


## Step 3-A: define the functions for transformation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
collate_fn = encode_transform_batch(vocabulary, device)


## Take a small batch

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)




## Step 4: batching the datasets

batch_size = 32  

train_dl = DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, collate_fn=collate_fn)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn)
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_fn)


# ### Embedding layers for sentence encoding
# 
# 
#  * `input_dim`: number of words, i.e. maximum integer index + 1.
#  * `output_dim`: 
#  * `input_length`: the length of (padded) sequence
#     * for example, `'This is an example' -> [0, 0, 0, 0, 0, 0, 3, 1, 8, 9]`   
#     => input_lenght is 10
#  
#  
# 
#  * When calling the layer, takes integr values as input,   
#  the embedding layer convert each interger into float vector of size `[output_dim]`
#    * If input shape is `[BATCH_SIZE]`, output shape will be `[BATCH_SIZE, output_dim]`
#    * If input shape is `[BATCH_SIZE, 10]`, output shape will be `[BATCH_SIZE, 10, output_dim]`







embedding = nn.Embedding(num_embeddings=10, 
                         embedding_dim=3, 
                         padding_idx=0)
 
# a batch of 2 samples of 4 indices each
text_encoded_input = torch.LongTensor([[1,2,4,5],[4,3,2,0]])
print(embedding(text_encoded_input))


# ### Building an RNN model
# 
# * **RNN layers:**
#   * `nn.RNN(input_size, hidden_size, num_layers=1)`
#   * `nn.LSTM(..)`
#   * `nn.GRU(..)`
#   * `nn.RNN(input_size, hidden_size, num_layers=1, bidirectional=True)`
#  
#  



## An example of building a RNN model
## with simple RNN layer

# Fully connected neural network with one hidden layer
model = RNN(64, 32) 
# model = model.to(device)

print(model) 
 
model(torch.randn(5, 3, 64)) 
summary(model, input_size=(1, 64), device="cpu")


# ### Building an RNN model for the sentiment analysis task
vocab_size = len(vocabulary)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64*2

torch.manual_seed(1)
loss_fn = nn.BCELoss()
model = embedRNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, loss_fn) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.optimizer = optimizer
model = model.to(device)

print(model)
summary(model, input_data=[text_batch, length_batch], verbose=2)

num_epochs = 10
torch.manual_seed(1)
model.train_epochs(num_epochs, train_dl, valid_dl, valid_size)



acc_test, _ = model.evaluate_procedure(test_dl, test_size)
print(f'test_accuracy: {acc_test:.4f}') 


# #### More on the bidirectional RNN

#  * **Trying bidirectional recurrent layer**

    
torch.manual_seed(1)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model = embedRNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, loss_fn, bidirectional=True) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model.optimizer = optimizer
model = model.to(device)

print(model)
summary(model, input_data=[text_batch, length_batch], verbose=2)

num_epochs = 10
torch.manual_seed(1)
model.train_epochs(num_epochs, train_dl, valid_dl, valid_size)

test_dataset = IMDB(split='test')
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_fn)


acc_test, _ = model.evaluate_procedure(test_dl, test_size)
print(f'test_accuracy: {acc_test:.4f}') 


# ## Optional exercise: 
# 
# ### Uni-directional SimpleRNN with full-length sequences

# 
# ---
torch.manual_seed(1)
loss_fn = nn.BCELoss()
model = embedRNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size, loss_fn,  rnn_type="simple")  
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model.optimizer = optimizer
model.to(device)

print(model) 
summary(model, input_data=[text_batch, length_batch], verbose=2)

num_epochs = 10
torch.manual_seed(1)
model.train_epochs(num_epochs, train_dl, valid_dl, valid_size)

test_dataset = IMDB(split='test')
test_dl = DataLoader(test_dataset, batch_size=batch_size,
                     shuffle=False, collate_fn=collate_fn)


acc_test, _ = model.evaluate_procedure(test_dl, test_size)
print(f'test_accuracy: {acc_test:.4f}') 

# 
# 
# Readers may ignore the next cell.
# 
