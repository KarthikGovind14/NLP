

def hyper_parameter():
    hyperparams = {
        "TrainBatchSize": 144,
        "ValidBatchSize": 160,
        "TestBatchSize": 160,
        "LearningRate": 0.01,
        "EmbeddingSize": 600,
        "HiddenSize": 800,
        "WeightDecay": 0.01,
        "Clip": 10,
        "Epsilon": 1e-7,
        "NEpochs": 50,
        "BPTT": 35,
        "Patience": 5,
        
    }
    return hyperparams

hyperparams = hyper_parameter()
