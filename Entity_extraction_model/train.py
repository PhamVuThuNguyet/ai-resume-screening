import pandas as pd
import numpy as np

import joblib
import torch
import ultis

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


if __name__ == "__main__":

    df_data = ultis.get_data(config.TRAINING_FILE)

    sentences, tag, enc_tag = process_data(df_data)
    meta_data = {
        "enc_tag": enc_tag
    }
    joblib.dump(meta_data, "meta.bin")
    num_tag = len(list(enc_tag.classes_))
    (train_sentences, test_sentences, train_tag, test_tag) = model_selection.train_test_split(sentences,
                                                                                              tag,
                                                                                              random_state=11,
                                                                                              test_size=0.2)

    train_dataset = EntityDataset(
        texts=train_sentences, tags=train_tag)
    valid_dataset = EntityDataset(
        texts=test_sentences, tags=test_tag)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, num_workers=4)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size, num_workers=2)

    device = torch.device("cuda")
    model = EntityModel(enc_tag=enc_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.1
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0
        }
    ]

    num_train_steps = int(len(train_sentences) / train_batch_size * epochs)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    best_loss = np.inf

    for epoch in range(epochs):

        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = eval_fn(valid_data_loader, model, device)
        print(f"Epoch: {epoch} Train Loss: {train_loss}, Test Loss: {test_loss}")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_path)