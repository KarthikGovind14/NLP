
from config import hyperparams
import torch, math, copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from functools import partial
import torch.optim as optim
from collections import Counter
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


Device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_data(path, eos_token = "<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output

class Lang():
    def __init__(self, corpus, special_tokens = []):
        # Create a vocabulary mapping from words to ids
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}
        
    def get_vocab(self, corpus, special_tokens = []):
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq

        padded_seqs = padded_seqs.detach() 
        return padded_seqs, lengths

    # Sort the data by the length of source sequences in descending order
    data.sort(key = lambda x: len(x["source"]), reverse = True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Merge and pad source and target sequences
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(Device)
    new_item["target"] = target.to(Device)
    new_item["number_tokens"] = sum(lengths)

    return new_item    

class PennTreeBank (data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])
            self.target.append(sentence.split()[1:])
        
        # Convert words to corresponding IDs using the provided language vocabulary
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res    