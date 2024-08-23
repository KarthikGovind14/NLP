from utils import *
from functions import *

# Function to compute cosine similarity between two vectors
def cosine_similarity(x, y):
    x_norm = norm(x)
    y_norm = norm(y)
    denom = x_norm * y_norm
    num = np.dot(x, y)
    return num / denom


class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index = 0, out_dropout = 0.3, emb_dropout = 0.3, n_layers = 1,
                 dropout = True, weightTying = False, variationalDropout = False):
        super(LM_LSTM, self).__init__()
        self.dropout = dropout
        self.variationalDropout = variationalDropout

        # Adjust hidden_size if WeightTying is enabled
        if weightTying:
            hidden_size = emb_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx = pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional = False)
        if self.dropout:
            if self.variationalDropout:
                self.embedding_dropout = VariationalDropout(emb_dropout)
                self.output_dropout = VariationalDropout(out_dropout)
            else:
                self.embedding_dropout = nn.Dropout(emb_dropout)
                self.output_dropout = nn.Dropout(out_dropout)

        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

        # If weight tying is enabled, set output layer's weight to be the same as the embedding layer's weight
        if weightTying:
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.dropout:
            emb = self.embedding_dropout(emb)
        
        lstm_output, _ = self.lstm(emb)
        if self.dropout:
            lstm_output = self.output_dropout(lstm_output)

        # Pass LSTM output through the linear output layer and permute dimensions for correct shape
        output = self.output(lstm_output).permute(0, 2, 1)
        return output
    
    # Get word embedding for a given token
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    # Find most similar words to a given vector using cosine similarity
    def get_most_similar(self, vector, top_k = 10):
        embs = self.embedding.weight.detach().cpu().numpy()
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))

        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]

        return (indexes, top_scores)

class VariationalDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training:
            return x

        # Apply dropout mask during training
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x

# Define a class for a custom optimizer Non-monotonically Triggered AvSGD
class NTASGDoptim(optim.Optimizer):
    def __init__(self, params, lr = 1, n = 5):
        defaults = dict(lr = lr, n = n, t0 = 10e7, t = 0, logs = [])
        super(NTASGDoptim, self).__init__(params, defaults)

    # Check for non-monotonic condition in the learning rate update
    def check(self, v):
        group = self.param_groups[0]
        if group['t'] > group['n'] and v > min(group['logs'][:-group['n']]):
            group['t0'] = self.state[next(iter(group['params']))]['step']
            print("Non-monotonic condition is triggered!")
            return True
        group['logs'].append(v)
        group['t'] += 1

    # Set the learning rate for the optimizer
    def lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
                               
    def step(self):
        group = self.param_groups[0]
        for p in group['params']:
            grad = p.grad.data
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['mu'] = 1
                state['ax'] = torch.zeros_like(p.data)
            state['step'] += 1
            
            p.data.add_(other = grad, alpha = -group['lr'])
            if state['mu'] != 1:
                state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
            else:
                state['ax'].copy_(p.data)

            state['mu'] = 1 / max(1, state['step'] - group['t0'])