from functions import *

if __name__ == "__main__":
    dataloaders, lang = get_dataloaders('dataset/ATIS/train.json', 'dataset/ATIS/test.json')

    # train(dataloaders, lang, bidirectional = True)                  # BATCH = 128/64/64 | EMB_SIZE = 300 | HIDDEN_SIZE = 200 | LR = 0.001
    # train(dataloaders, lang, bidirectional = True, dropout = True)  # BATCH = 128/64/64 | EMB_SIZE = 300 | HIDDEN_SIZE = 200 | LR = 0.001 | DROPOUT = 0.3

    train(dataloaders, lang, trained = True, bidirectional = True)
    train(dataloaders, lang, trained = True, dropout = True)
