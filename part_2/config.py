from transformers import BertModel, BertTokenizer

def hyper_parameter():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    hyperparams = {
        "TrainBatchSize": 64,
        "ValidBatchSize": 80,
        "TestBatchSize": 80,
        "LearningRate": 0.005,
        "Dropout": 0.1,
        "Patience": 20,
        "NEpochs": 400,
        "OutputSlot": lambda x: len(x.slot2id),
        "OutputIntent": lambda x: len(x.intent2id),
        "VocabularyLength": lambda x: len(x.word2id),
        "PadToken": tokenizer.pad_token_id,  # BERT padding token
        "ClsToken": tokenizer.cls_token_id,  # BERT CLS token
        "SepToken": tokenizer.sep_token_id,  # BERT SEP token
        "UnkToken": tokenizer.unk_token_id   # BERT UNK token
    }
    return hyperparams

hyperparams = hyper_parameter()