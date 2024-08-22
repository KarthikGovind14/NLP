

def hyper_parameter():
    hyperparams = {
        "TrainBatchSize": 64,
        "ValidBatchSize": 32,
        "TestBatchSize": 32,
        "EmbeddingSize": 500,
        "HiddenSize": 300,
        "LearningRate": 0.01,
        "Dropout": 0.2,
        "Patience": 10,
        "NEpochs": 10,
        "OutputSlot": lambda x: len(x.slot2id),
        "OutputIntent": lambda x: len(x.intent2id),
        "VocabularyLength": lambda x: len(x.word2id),
        "PadToken": 0
    }
    return hyperparams

hyperparams = hyper_parameter()
