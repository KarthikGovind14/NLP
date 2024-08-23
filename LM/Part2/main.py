from functions import *
from config import hyperparams
from utils import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim

def main(trained = False,  weightTying = False, VariationalDropout = False, NTASGD = False):

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
        if NTASGD:
            model_name = "Lstm Model With Variational Dropout And Weight Tying Using Ntasgd Optimizer"

        elif VariationalDropout:
            model_name = "Lstm Model With Variational Dropout And Weight Tying Using Adamw Optimizer"
            
        else:
            model_name = "Lstm Model With Dropout And Weight Tying Using Adamw Optimizer"
        
        print("\n" + "=" * (len(model_name) + 2))
        print(model_name)
        print("=" * (len(model_name) + 2) + "\n")
        
        vocab_len = len(lang.word2id)
        model = LM_LSTM(hyperparams["EmbeddingSize"], hyperparams["HiddenSize"], vocab_len, pad_index = lang.word2id["<pad>"], weightTying = weightTying, variationalDropout = VariationalDropout).to(Device)
        model.apply(init_weights)

        if NTASGD:
            optimizer = NTASGDoptim(model.parameters(), lr = hyperparams["LearningRate"])
        else:
            optimizer = optim.AdamW(model.parameters(), lr = hyperparams["LearningRate"], weight_decay = hyperparams["WeightDecay"], eps = hyperparams["Epsilon"])

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_ppl = math.inf
        best_model = None
        pbar = tqdm(range(1, hyperparams["NEpochs"]))
        patience = hyperparams["Patience"]

        for epoch in pbar:
            if NTASGD:
                seq_len = get_seq_len(hyperparams["BPTT"])
                optimizer.lr(seq_len / hyperparams["BPTT"] * hyperparams["LearningRate"])

            loss = train_loop(train_loader, optimizer, criterion_train, model, hyperparams["Clip"])
            print()
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())

                if NTASGD:
                    tmp = {}
                    for (prm,st) in optimizer.state.items():
                        tmp[prm] = prm.clone().detach()
                        prm.data = st['ax'].clone().detach()

                ppl_valid, loss_valid = eval_loop(valid_loader, criterion_eval, model)

                if NTASGD:
                    optimizer.check(ppl_valid)

                losses_valid.append(np.asarray(loss_valid).mean())
                pbar.set_description("Train PPL: %f" % ppl_valid)
                if ppl_valid < best_ppl:
                    best_ppl = ppl_valid
                    best_model = copy.deepcopy(model).to(Device)
                    patience = hyperparams["Patience"]
                else:
                    patience -= 1
                
                if NTASGD:
                    for (prm,st) in optimizer.state.items():
                        prm.data = tmp[prm].clone().detach()

                if patience <= 0:
                    break

        best_model.to(Device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        if NTASGD:
            torch.save(best_model, 'bin/LSTM_WT_VD_NTASGD.pt')
        elif VariationalDropout:
            torch.save(best_model, 'bin/LSTM_WT_VD_AdamW.pt')
        else:
            torch.save(best_model, 'bin/LSTM_WT_DP_AdamW.pt')
            
    # Loading a pre-trained model
    else:
        if NTASGD:
            model_name = "Lstm Model With Variational Dropout And Weight Tying Using Ntasgd Optimizer"
            best_model = torch.load('bin/LSTM_WT_VD_NTASGD.pt', map_location=Device, weights_only=False)
        elif VariationalDropout:
            model_name = "Lstm Model With Variational Dropout And Weight Tying Using Adamw Optimizer"
            best_model = torch.load('bin/LSTM_WT_VD_AdamW.pt', map_location=Device, weights_only=False)
        else:
            model_name = "Lstm Model With Dropout And Weight Tying Using Adamw Optimizer"
            best_model = torch.load('bin/LSTM_WT_DP_AdamW.pt', map_location=Device, weights_only=False)
        
        # Print the model name with dynamic separator length
        print("\n" + "=" * (len(model_name) + 2))
        print(model_name)
        print("=" * (len(model_name) + 2) + "\n")
        
        print("Best Trained Model Loaded")
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        
        print("\nTest PPL: ", final_ppl, "\n")



if __name__ == "__main__":

    main(trained=True, weightTying = True)

    main(trained=True, VariationalDropout = True)

    main(trained=True, NTASGD = True)
    