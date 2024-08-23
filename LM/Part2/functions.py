from utils import *
from model import *
from config import hyperparams


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for sample in data:
            output = model(sample['source'])  # Forward pass
            loss = eval_criterion(output, sample['target'])  # Compute loss
            loss_array.append(loss.item())  # Accumulate loss
            number_of_tokens.append(sample["number_tokens"])  # Accumulate token count

    # Compute perplexity
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def train_loop(data, optimizer, criterion, model, clip):
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:
        optimizer.zero_grad()  # Clear gradients
        output = model(sample["source"])  # Forward pass
        loss = criterion(output, sample["target"])  # Compute loss
        loss_array.append(loss.item() * sample["number_tokens"])  # Accumulate loss
        number_of_tokens.append(sample["number_tokens"])  # Accumulate token count
        loss.backward()  # Backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Clip gradients to prevent exploding gradients
        optimizer.step()   # Update model parameters

    return sum(loss_array) / sum(number_of_tokens)

def init_weights(mat):
    """
    Initialize the weights of the given module.

    Args:
        mat (torch.nn.Module): Module to initialize weights for.
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def get_seq_len(bptt):
        seq_len = bptt if np.random.random() < 0.95 else bptt/2
        seq_len = round(np.random.normal(seq_len, 5))
        while seq_len <= 5 or seq_len >= 90:
            seq_len = bptt if np.random.random() < 0.95 else bptt/2
            seq_len = round(np.random.normal(seq_len, 5))
        return seq_len

def build_dataloaders(train_raw, dev_raw, test_raw, lang):
    """
    Build dataloaders for training, validation, and testing.


    Returns:
        tuple: Train, validation, and test dataloaders.
    """
    train_dataset = PennTreeBank(train_raw, lang)
    valid_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams['TrainBatchSize'], collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]), shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size=hyperparams['ValidBatchSize'], collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['TestBatchSize'], collate_fn = partial(collate_fn, pad_token = lang.word2id["<pad>"]))

    return train_loader, valid_loader, test_loader
