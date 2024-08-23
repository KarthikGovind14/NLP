from functions import *
from config import hyperparams
from utils import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim


def main(trained=False):


    Device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_file, test_file = 'dataset/ATIS/train.json', 'dataset/ATIS/test.json'

    tmp_train_raw = load_data(train_file)
    test_raw = load_data(test_file)

    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10) / (len(tmp_train_raw)), 2)
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_Train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])

    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, random_state=42, stratify=labels)
    X_train.extend(mini_Train)
    train_raw = X_train
    valid_raw = X_dev

    # Create language data structure
    lang = get_lang(train_raw, valid_raw, test_raw)

    dataLoader = build_dataloaders(train_raw, valid_raw, test_raw, lang)

    train_loader, valid_loader, test_loader = dataLoader
    criterion_slots = nn.CrossEntropyLoss(ignore_index=hyperparams["PadToken"])
    criterion_intents = nn.CrossEntropyLoss()

    if not trained:
        model_name = "BERT Model"
        print("\n" + "=" * (len(model_name) + 2))
        print(model_name)
        print("=" * (len(model_name) + 2) + "\n")

        model = BERT(hyperparams["OutputSlot"](lang), hyperparams["OutputIntent"](lang), dropout=hyperparams["Dropout"]).to(Device)

        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=hyperparams["LearningRate"])

        losses_train = []
        losses_valid = []
        sampled_epochs = []
        best_f1 = 0
        patience = hyperparams["Patience"]

        for X in tqdm(range(1, hyperparams["NEpochs"])):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            if X % 5 == 0:
                sampled_epochs.append(X)
                losses_train.append(np.mean(loss))
                results_valid, _, loss_valid = eval_loop(valid_loader, criterion_slots, criterion_intents, model, lang)
                losses_valid.append(np.mean(loss_valid))
                f1 = results_valid["total"]["f"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model)
                    patience = hyperparams["Patience"]
                else:
                    patience -= 1

                if patience <= 0:
                    break

        model.to(Device)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        torch.save(model, 'bin/BERT.pt')

    else:
        model_name = "BERT Model"
        print("\n" + "-" * (len(model_name) + 2))
        print(model_name)
        print("-" * (len(model_name) + 2) + "\n")

        model_path = 'bin/BERT.pt'
        best_model = torch.load(model_path, map_location=Device, weights_only = False)

        print("Best Model")
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)

    print("\nSlot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"], "\n")

if __name__ == "__main__":

    main(trained=True)