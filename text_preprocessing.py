import re
import torch
import torch.nn as nn
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
import torchtext
from torchtext import __version__ as torchtext_version
from pkg_resources import parse_version


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


def vocab_builder(token_counts):
    sorted_by_freq_tuples = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    vocabulary = vocab(ordered_dict)

    vocabulary.insert_token("<pad>", 0)
    vocabulary.insert_token("<unk>", 1)
    vocabulary.set_default_index(1)

    return vocabulary


## Step 3-B: wrap the encode and transformation function
def encode_transform_batch(vocabulary, device):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_pipeline = lambda x: [vocabulary[token] for token in tokenizer(x)]
    if parse_version(torchtext.__version__) > parse_version("0.10"):
        label_pipeline = lambda x: 1. if x == 2 else 0.         # 1 ~ negative, 2 ~ positive review
    else:
        label_pipeline = lambda x: 1. if x == 'pos' else 0.

    def collate_fn(batch):
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), 
                                        dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(processed_text.size(0))
        label_list = torch.tensor(label_list)
        lengths = torch.tensor(lengths)
        padded_text_list = nn.utils.rnn.pad_sequence(
            text_list, batch_first=True)
        return padded_text_list.to(device), label_list.to(device), lengths.to(device)
    
    return collate_fn