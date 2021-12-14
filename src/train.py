import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import joblib

from sklearn import preprocessing
from sklearn import model_selection

from transformers import get_linear_schedule_with_warmup

import model
import utils
import config
import dataset
import engine

def process_data(data_path):
    df = pd.read_csv(data_path,encoding='latin-1')
    df = df[:5000]
    df.loc[:,"Sentence #"] = df["Sentence #"].fillna(method="ffill")

    word_to_ix = {}
    tag_to_ix = {}

    word_to_ix["_pad_"] = 0
    tag_to_ix["_pad_"] = 0

    for _,row in df.iterrows():
        word = row["Word"]
        tag = row["Tag"]

        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix) 

    tag_to_ix[config.START_TAG] = len(tag_to_ix)
    tag_to_ix[config.STOP_TAG] = len(tag_to_ix)

    sentences = df.groupby("Sentence #")["Word"].apply(list).values 
    tag_sequences = df.groupby("Sentence #")["Tag"].apply(list).values

    enc_sentences = []  
    for sentence in sentences:
        idxs = [word_to_ix[w] for w in sentence]
        enc_sentences.append(idxs)

    enc_tags = []
    for tag_seq in tag_sequences:
        tag_idxs = [tag_to_ix[tag] for tag in tag_seq]
        enc_tags.append(tag_idxs)

    return enc_sentences, enc_tags, word_to_ix, tag_to_ix


if __name__ == '__main__':
    torch.manual_seed(1)

    sentences, tag,  word_to_ix, tag_to_ix = process_data(config.TRAINING_FILE)


    num_tag = len(tag_to_ix)

    (
        train_sentences, 
        test_sentences, 
        train_tag,
        test_tag
     ) = model_selection.train_test_split(sentences,tag,random_state = 42,test_size = 0.1)

    training_data = {'Sentence': train_sentences, 'Tag': train_tag}
    training_data = pd.DataFrame(training_data)

    test_data = {'Sentence': test_sentences, 'Tag': test_tag}
    test_data = pd.DataFrame(test_data)

    device = torch.device("cuda")
    model = model.BiLSTM_CRF(len(word_to_ix), tag_to_ix, config.EMBEDDING_DIM, config.HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, weight_decay = 1e-4)
    model.to(device)

    num_train_steps = int(len(train_sentences)/config.TRAIN_BATCH_SIZE*config.EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps = num_train_steps
    )

    # with torch.no_grad():
    #     precheck_sent = utils.prepare_sequence(training_data[0][0], word_to_ix)
    #     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype = torch.long)
    #     print(model(precheck_sent))

    for epoch in range(10):
        print("epoch: "+str(epoch))
        total_accuracy = 0.0
        total_loss = 0.0
        num = len(training_data.index)
        for index, row in training_data.iterrows():


            sentence = row["Sentence"]
            tag = row["Tag"]

            
            model.zero_grad()
            sentence_in = torch.tensor(sentence, dtype = torch.long).to(device)
            targets = torch.tensor(tag, dtype = torch.long).to(device)

            loss,accuracy_score = model.neg_log_likelihood(sentence_in, targets)

            total_accuracy += accuracy_score
            total_loss += loss.item()
            if(index %100 ==0):
                print("index = "+str(index))
                print("loss = "+ str(loss.item()))

            loss.backward()
            optimizer.step()
        print(f"loss = {total_loss/num}; accuracy = {total_accuracy/num}")


    with torch.no_grad():
        precheck_sent = [317, 142, 274, 127, 109, 318, 18, 4, 308, 12, 10, 319, 320, 14, 321, 310, 94, 252, 322, 323, 117, 25, 54, 287, 324, 325, 14, 326, 297, 298, 327, 22]
        print(model(precheck_sent)) 
