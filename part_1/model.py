from utils import *

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer = 1, pad_index = 0, bidirectional = False, dropout = None):
        super(ModelIAS, self).__init__()
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx = pad_index)  # Embedding layer to convert word indices to dense vectors
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional = bidirectional)  # Utterance encoder using LSTM
        self.slot_out = nn.Linear(hid_size * (2 if bidirectional else 1), out_slot)  # Output layer for slot prediction
        self.intent_out = nn.Linear(hid_size * (2 if bidirectional else 1), out_int)  # Output layer for intent prediction
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)  # Convert word indices to embedded representations
        utt_emb = utt_emb.permute(1, 0, 2)  # Permute the dimensions to fit LSTM input shape (sequence_length, batch_size, embedding_size)
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())  # Pack the padded sequence to optimize computation
        packed_output, (last_hidden, _) = self.utt_encoder(packed_input)  # Forward pass through the LSTM
        utt_encoded, _ = pad_packed_sequence(packed_output)  # Unpack the packed output sequence

        # Concatenate hidden states from both directions if bidirectional
        last_hidden = torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim = 1)
        if self.dropout:
            utt_encoded = self.dropout(utt_encoded)
            last_hidden = self.dropout(last_hidden)

        slots = self.slot_out(utt_encoded)  # Generate slot predictions for each time step in the sequence
        intent = self.intent_out(last_hidden)  # Generate intent prediction based on the final hidden state
        slots = slots.permute(1, 2, 0)  # Permute slot predictions for compatibility (batch_size, num_slots, sequence_length)

        return slots, intent