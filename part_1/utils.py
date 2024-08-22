from conll import *
from config import hyperparams
import torch, json, copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


Device = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_data(path):
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_vocabulary(train_raw, dev_raw, test_raw):
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff = 0)

    return lang

class Lang:
        # Create word-to-id, slot-to-id, and intent-to-id mappings
    def __init__(self, words, intents, slots, cutoff = 0):
        self.word2id = self.w2id(words, cutoff = cutoff, unk = True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad = False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    # Create word-to-id mapping
    def w2id(self, elements, cutoff = None, unk = True, load = True):
        # Load vocabulary mapping if available, otherwise create and save
        if load:
            try:
                with open("dataset/w2id.json", "r") as f:
                    vocab = json.load(f)
            except FileNotFoundError:
                vocab = self.w2id(elements, cutoff = cutoff, unk = unk, load = False)
            return vocab
        else:
            vocab = {"pad": hyperparams['PadToken']}
            if unk: vocab["unk"] = len(vocab)
            count = Counter(elements)
            for k, v in count.items():
                if v > cutoff: vocab[k] = len(vocab)
            with open("dataset/w2id.json", "w") as f:
                json.dump(vocab, f)
            return vocab

    # Create label-to-id mapping
    def lab2id(self, elements, pad = True, load = True):
        # Load label mapping if available, otherwise create and save
        if load:
            try:
                if pad:
                    with open("dataset/lab2id.json", "r") as f:
                        vocab = json.load(f)
                else:
                    with open("dataset/intent2id.json", "r") as f:
                        vocab = json.load(f)
            except FileNotFoundError:
                vocab = self.lab2id(elements, pad = pad, load = False)
            finally:
                return vocab
        else:
            vocab = {}
            if pad: vocab["pad"] = hyperparams['PadToken']
            for elem in elements:
                vocab[elem] = len(vocab)
            if pad:
                with open("dataset/lab2id.json", "w") as f:
                    json.dump(vocab, f)
            else:
                with open("dataset/intent2id.json", "w") as f:
                    json.dump(vocab, f)
            return vocab


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(hyperparams['PadToken'])
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        padded_seqs = padded_seqs.detach()
        return padded_seqs, lengths
    
    # Sort the data by utterance length
    data.sort(key = lambda x: len(x['utterance']), reverse = True)
    new_item = {}
    # Reorganize the data into batches
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(Device)
    y_slots = y_slots.to(Device)
    intent = intent.to(Device)
    y_lengths = torch.LongTensor(y_lengths).to(Device)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk = 'unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # Map data to corresponding IDs
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Map labels to IDs
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    # Map sequences to IDs
    def mapping_seq(self, data, mapper):
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res