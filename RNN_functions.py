import numpy as np
import pandas as pd
import torch
from torchtext import data
from RNN import LSTM
from sklearn import metrics
import warnings


def prepare_data():

   # load data
    with open('q1_train_cleaned.txt') as f:
        content = f.readlines()
    q1_train = [x.strip() for x in content]

    with open('q2_train_cleaned.txt') as f:
        content = f.readlines()
    q2_train = [x.strip() for x in content]

    with open('q1_test_cleaned.txt') as f:
        content = f.readlines()
    q1_test = [x.strip() for x in content]

    with open('q2_test_cleaned.txt') as f:
        content = f.readlines()
    q2_test = [x.strip() for x in content]

    with open('test_labels.txt') as f:
        content = f.readlines()
    labels_test = [x.strip() for x in content]

    with open('train_labels.txt') as f:
        content = f.readlines()
    labels_train = [x.strip() for x in content]

    # save data as csv
    train_df = pd.DataFrame(np.stack((q1_train, q2_train, labels_train),
                                     axis=-1), columns=['question1', 'question2', 'is_duplicate'])
    test_df = pd.DataFrame(np.stack((q1_test, q2_test, labels_test),
                                    axis=-1), columns=['question1', 'question2', 'is_duplicate'])
    train_df.to_csv('train_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)

    # read data and prepare fields
    question1 = data.Field(tokenize='spacy')
    question2 = data.Field(tokenize='spacy')
    label = data.LabelField(dtype=torch.float)

    fields = [('question1', question1), ('question2',
                                         question2), ('is_duplicate', label)]

    train_data, test_data = data.TabularDataset.splits(
        path='.', train='train_df.csv', test='test_df.csv', format='csv', fields=fields, skip_header=True)

    # build vocabularies (same for question1 and question2)
    MAX_VOCAB_SIZE = 60000

    # train data
    question1.build_vocab(train_data.question1, train_data.question2,
                          max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    question2.build_vocab(train_data.question1, train_data.question2,
                          max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)

    # test data
    question1.build_vocab(test_data.question1, test_data.question2,
                          max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    question2.build_vocab(test_data.question1, test_data.question2,
                          max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    label.build_vocab(test_data, max_size=MAX_VOCAB_SIZE)

    # prepare iterators
    BATCH_SIZE = 64

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        sort=False,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False)
    return train_iterator, test_iterator, question1


def train(train_iterator, test_iterator, question1):
    warnings.filterwarnings("ignore")

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model to device
    model = LSTM(vocab_dim=len(question1.vocab), text=question1)
    model.to(device)

    epochs = 20
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1*10**(-3))
    softmax = torch.nn.Softmax()

    output_values = []
    labels_values = []
    auc_old = 0

    #train and test
    for epoch in range(1, epochs+1):
        loss_values = []

        for batch in train_iterator:
            output = model(batch.question1.to(device),
                           batch.question2.to(device))
            loss = criterion(output, batch.is_duplicate.long().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
        print(epoch, np.mean(loss_values))

        for batch in test_iterator:
            output = model(batch.question1.to(device),
                           batch.question2.to(device))
            output = torch.argmax(softmax(output), dim=1)
            output_values = np.concatenate(
                (output_values, output.cpu().detach().numpy()))
            labels_values = np.concatenate(
                (labels_values, batch.is_duplicate.cpu().detach().numpy()))
        auc_new = metrics.roc_auc_score(output_values, labels_values)
        print(epoch, auc_new)

        if auc_old < auc_new:
            # save the model if it is improved
            torch.save(model.state_dict(), "model.pth")
        auc_old = auc_new
    return

    def predict(test_iterator, question1):
        softmax = torch.nn.Softmax()
        output_values = []
        labels_values = []

        # model to device
        model = LSTM(vocab_dim=len(question1.vocab), text=question1)
        model.load_state_dict(torch.load('model.pth'))
        model.to(device)

        # final test with the best model
        for batch in test_iterator:
            output = model(batch.question1.to(device),
                           batch.question2.to(device))
            output = torch.argmax(softmax(output), dim=1)
            output_values = np.concatenate(
                (output_values, output.cpu().detach().numpy()))
            labels_values = np.concatenate(
                (labels_values, batch.is_duplicate.cpu().detach().numpy()))
        auc = metrics.roc_auc_score(output_values, labels_values)
        print(auc)
    return
