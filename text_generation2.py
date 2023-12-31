# coding: utf-8


import sys
from python_environment_check import check_packages
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions.categorical import Categorical
import time

# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# ## Package version checks

# Add folder to path in order to load from the check_packages.py script:



sys.path.insert(0, '..')


# Check recommended package versions:

d = {
    'torch': '1.8.0',
}
check_packages(d)


# Chapter 15: Modeling Sequential Data Using Recurrent Neural Networks (part 3/3)
# ## Project two: character-level language modeling in PyTorch
# 


# ### Preprocessing the dataset

## Reading and processing text
with open('1268-0.txt', 'r', encoding="utf8") as fp:
    text=fp.read()
    
start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')

text = text[start_indx:end_indx]
char_set = set(text)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))



chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32)

print('Text encoded shape: ', text_encoded.shape)

print(text[:15], '     == Encoding ==> ', text_encoded[:15])
print(text_encoded[15:21], ' == Reverse  ==> ', ''.join(char_array[text_encoded[15:21]]))




for ex in text_encoded[:5]:
    print('{} -> {}'.format(ex, char_array[ex]))



seq_length = 40
chunk_size = seq_length + 1

text_chunks = [text_encoded[i:i+chunk_size] 
               for i in range(len(text_encoded)-chunk_size+1)] 

## inspection:
for seq in text_chunks[:1]:
    input_seq = seq[:seq_length]
    target = seq[seq_length] 
    print(input_seq, ' -> ', target)
    print(repr(''.join(char_array[input_seq])), 
          ' -> ', repr(''.join(char_array[target])))



class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()
    
seq_dataset = TextDataset(torch.tensor(text_chunks))




for i, (seq, target) in enumerate(seq_dataset):
    print(' Input (x):', repr(''.join(char_array[seq])))
    print('Target (y):', repr(''.join(char_array[target])))
    print()
    if i == 1:
        break


device = torch.device("cuda:0")
# device = 'cpu'


batch_size = 64

torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# ### Building a character-level RNN model

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, 
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(device), cell.to(device)
    
vocab_size = len(char_array)
embed_dim = 256
# embed_dim = 80
rnn_hidden_size = 512
# rnn_hidden_size = 512*2

torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size) 
model = model.to(device)
model




loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=)

# num_epochs = 10000 
num_epochs = 100

torch.manual_seed(1)

def train_procedure(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    counter = 0
    for seq_batch, target_batch in dataloader:
        seq_batch = seq_batch.to(device)
        target_batch = target_batch.to(device)
        optimizer.zero_grad()
        loss = 0
        hidden, cell = model.init_hidden(batch_size)
        for c in range(seq_length):
            pred, hidden, cell = model(seq_batch[:, c], hidden, cell) 
            loss += loss_fn(pred, target_batch[:, c])
        loss.backward()
        optimizer.step()
        loss = loss.item()/seq_length
        total_loss += loss
        counter += 1
    return total_loss/counter
    #     total_acc += ((pred>=0.5).float() == label_batch).float().sum().item()
    #     total_loss += loss.item()*label_batch.size(0)
    # return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

TRAIN = False
LOAD = True
SAVE = False
PATH = './models/text_generation2/text_generation2.pt'
if LOAD:
    model = torch.load(PATH)
    if TRAIN:
        model.train()
    else:
        model.eval()
if TRAIN:
    start_time = time.time()
    for epoch in range(num_epochs):
        loss = train_procedure(seq_dl)
        if epoch % 1 == 0:
            end_time = time.time()
            print(f'Epoch {epoch} loss: {loss:.4f} time: {(end_time - start_time):.3f} secs')
            start_time = time.time()
            # Save learned model
            if SAVE:
                torch.save(model, PATH)


model.eval()

# ### Evaluation phase: generating new text passages

torch.manual_seed(1)

logits = torch.tensor([[1.0, 1.0, 1.0]])

print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])

m = Categorical(logits=logits)
samples = m.sample((10,))
 
print(samples.numpy())




torch.manual_seed(1)

logits = torch.tensor([[1.0, 1.0, 3.0]])

print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])

m = Categorical(logits=logits)
samples = m.sample((10,))
 
print(samples.numpy())




def sample(model, starting_str, 
           len_generated_text=500, 
           scale_factor=1.0):

    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden = hidden.to('cpu')
    cell = cell.to('cpu')
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell) 
    
    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell) 
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])
        
    return generated_str

torch.manual_seed(1)
model.to('cpu')
print(sample(model, starting_str='The island'))


# * **Predictability vs. randomness**



logits = torch.tensor([[1.0, 1.0, 3.0]])

print('Probabilities before scaling:        ', nn.functional.softmax(logits, dim=1).numpy()[0])

print('Probabilities after scaling with 0.5:', nn.functional.softmax(0.5*logits, dim=1).numpy()[0])

print('Probabilities after scaling with 0.1:', nn.functional.softmax(0.1*logits, dim=1).numpy()[0])




torch.manual_seed(1)
print(sample(model, starting_str='The island', 
             scale_factor=2.0))




torch.manual_seed(1)
print(sample(model, starting_str='The island', 
             scale_factor=0.5))





