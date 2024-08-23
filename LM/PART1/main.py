from functions import *
from config import hyperparams
from utils import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim

def main(trained = False, dropout = False, AdamW = False):

    Device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_file, valid_file, test_file = 'dataset/ptb.train.txt', 'dataset/ptb.valid.txt', 'dataset/ptb.test.txt'

    train_raw = load_data(train_file)
    dev_raw = load_data(valid_file)
    test_raw = load_data(test_file)

    # Create a language instance
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    dataLoader = build_dataloaders(train_raw, dev_raw, test_raw, lang)

    train_loader, valid_loader, test_loader = dataLoader
    criterion_train = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index = lang.word2id["<pad>"], reduction = "sum")

    if not trained:
        if AdamW:
            model_name = "LSTM MODEL WITH DROPOUT USING ADAMW OPTIMIZER"
        elif dropout:
            model_name = "LSTM MODEL WITH DROPOUT"
        else:
            model_name = "LSTM MODEL"
        
        print("\n" + "=" * (len(model_name) + 2))
        print(model_name)
        print("=" * (len(model_name) + 2) + "\n")
        
        vocab_len = len(lang.word2id)
        model = LM_LSTM(hyperparams["EmbeddingSize"], hyperparams["HiddenSize"], vocab_len, pad_index = lang.word2id["<pad>"], dropout = dropout).to(Device)
        model.apply(init_weights)

        if AdamW:
            optimizer = optim.AdamW(model.parameters(), lr = hyperparams["LearningRate"], weight_decay = hyperparams["WeightDecay"], eps = hyperparams["Epsilon"])
        else:
            optimizer = optim.SGD(model.parameters(), lr = hyperparams["LearningRate"])

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, hyperparams["NEpochs"]))
        patience = hyperparams["Patience"]

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, hyperparams["Clip"])
            print()
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)  # Store epoch for plotting
                losses_train.append(np.asarray(loss).mean())  # Store average training loss
                ppl_valid, loss_valid = eval_loop(valid_loader, criterion_eval, model)
                losses_valid.append(np.asarray(loss_valid).mean())  # Store average validation loss
                pbar.set_description("Train PPL: %f" % ppl_valid)  # Update progress bar description
                if ppl_valid < best_ppl:
                    best_ppl = ppl_valid
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = hyperparams["Patience"]
                else:
                    patience -= 1

                # Early stopping
                if patience <= 0:
                    break

        best_model.to(Device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        if AdamW:
            torch.save(best_model, 'bin/LSTM+dropout+AdamW.pt')
        elif dropout:
            torch.save(best_model, 'bin/LSTM+dropout.pt')
        else:
            torch.save(best_model, 'bin/LSTM.pt')
            
    # Loading a pre-trained model
    else:
        if AdamW:
            model_name = "Lstm Model With Dropout Using AdamW Optimizer"
            best_model = torch.load('bin/LSTM+dropout+AdamW.pt', weights_only=False, map_location=Device)
        elif dropout:
            model_name = "Lstm Model With Dropout"
            best_model = torch.load('bin/LSTM+dropout.pt', weights_only=False, map_location=Device)
        else:
            model_name = "Lstm Model"
            best_model = torch.load('bin/LSTM.pt', weights_only=False, map_location=Device)
        
        # Print the model name with dynamic separator length
        print("\n" + "=" * (len(model_name) + 2))
        print(model_name)
        print("=" * (len(model_name) + 2) + "\n")
        
        print("Best Trained Model Loaded")
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        
    print("\nTest PPL: ", final_ppl, "\n")



if __name__ == "__main__":

    main(trained=True)

    main(trained=True, dropout=True)

    main(trained=True, dropout=True, AdamW=True)
