from utils import *
from model import *

# Set the device to CUDA if available, otherwise use CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 200
LEARNING_RATE = 0.001
DROPOUT = 0.3
PATIENCE = 20
N_EPOCHS = 400
OUTPUT_SLOT = lambda x: len(x.slot2id)
OUTPUT_INTENT = lambda x: len(x.intent2id)
VOCABULARY_LENGHT = lambda x: len(x.word2id)

def get_dataloaders(train_file, test_file):
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

    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size = portion, random_state = 42, stratify = labels)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    # Create language data structure
    lang = get_vocabulary(train_raw, valid_raw, test_raw)

    train_dataset = IntentsAndSlots(train_raw, lang)
    valid_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size = TRAIN_BATCH_SIZE, collate_fn = collate_fn,  shuffle = True)
    valid_loader = DataLoader(valid_dataset, batch_size = VALID_BATCH_SIZE, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = TEST_BATCH_SIZE, collate_fn = collate_fn)

    return [train_loader, valid_loader, test_loader], lang

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_array

#checked
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()

    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim = 1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim = 1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        pass

    report_intent = classification_report(ref_intents, hyp_intents, zero_division = False, output_dict = True)
    return results, report_intent, loss_array

#checked
# Function to initialize weights in the neural network
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def get_model(lang, bidirectional, dropout):
    if dropout:
        model = ModelIAS(HIDDEN_SIZE, OUTPUT_SLOT(lang), OUTPUT_INTENT(lang), EMBEDDING_SIZE, VOCABULARY_LENGHT(lang), pad_index = PAD_TOKEN, 
                        bidirectional = bidirectional, dropout = DROPOUT)
    else:
        model = ModelIAS(HIDDEN_SIZE, OUTPUT_SLOT(lang), OUTPUT_INTENT(lang), EMBEDDING_SIZE, VOCABULARY_LENGHT(lang), pad_index = PAD_TOKEN, 
                        bidirectional = bidirectional)
    return model    

#checked

def train(dataloaders, lang, trained = False, bidirectional = False, dropout = False):
    train_loader, valid_loader, test_loader = dataloaders
    criterion_slots = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if not trained:
        if dropout:
            print("\n" + "="*76, "LSTM BIDIRECTIONAL MODEL WITH DROPOUT", "="*76 + "\n")
        elif bidirectional:
            print("\n" + "="*82, "LSTM BIDIRECTIONAL MODEL", "="*83 + "\n")
        else:
            return

        model = get_model(lang, bidirectional, dropout).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

        losses_train = []
        losses_valid = []
        sampled_epochs = [] 
        best_f1 = 0
        best_model = None
        patience = PATIENCE

        for x in tqdm(range(1, N_EPOCHS)):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            if x % 5 == 0:
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_valid, _, loss_valid = eval_loop(valid_loader, criterion_slots, criterion_intents, model, lang)
                losses_valid.append(np.asarray(loss_valid).mean())
                f1 = results_valid["total"]["f"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model)
                    patience = PATIENCE
                else:
                    patience -= 1

                if patience <= 0:
                    break

        best_model.to(DEVICE)
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
        if dropout:
            torch.save(best_model, 'bin/LSTM+bidirectional+dropout.pt')
        elif bidirectional:
            torch.save(best_model, 'bin/LSTM+bidirectional.pt')

    else:
        if dropout:
            print("\n" + "="*39, "LSTM BIDIRECTIONAL MODEL WITH DROPOUT", "="*38)
            best_model = torch.load('bin/LSTM+bidirectional+dropout.pt', map_location = DEVICE)
        elif bidirectional:
            print("\n" + "="*45, "LSTM BIDIRECTIONAL MODEL", "="*45)
            best_model = torch.load('bin/LSTM+bidirectional.pt', map_location = DEVICE)
        else:
            return

        print("BEST TRAINED MODEL")
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)

    print("\nSlot F1: ", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"], "\n")