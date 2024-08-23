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
    def __init__(self, emb_size, hidden_size, output_size, pad_index = 0, out_dropout = 0.3, emb_dropout = 0.3, n_layers = 1, dropout = False):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx = pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional = False)
        self.dropout = dropout
        if self.dropout:
            self.embedding_dropout = nn.Dropout(emb_dropout)
            self.output_dropout = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Perform embedding lookup for the input sequence
        emb = self.embedding(input_sequence)

        # Apply dropout to the embeddings if enabled
        if self.dropout:
            emb = self.embedding_dropout(emb)
        
        lstm_output, _ = self.lstm(emb)
        if self.dropout:
            lstm_output = self.output_dropout(lstm_output)

        # Apply linear transformation to get the output logits and permute dimensions
        output = self.output(lstm_output).permute(0, 2, 1)
        return output
    
    def get_word_embedding(self, token):
        # Get the embedding for a specific token
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k = 10):
        # Get the weight matrix of the embedding layer
        embs = self.embedding.weight.detach().cpu().numpy()
    
        # Calculate cosine similarity scores between the given vector and all embeddings
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))

        scores = np.asarray(scores)

        # Get indexes of top-k embeddings with highest similarity scores
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]

        return (indexes, top_scores)