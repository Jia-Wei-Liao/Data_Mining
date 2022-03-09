import os
import re
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import transformers

from utils import *
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


df = pd.read_csv('TrainData.tsv')
df.head()

mode = 'training'
pretrain = ''
sentence = 'sentence2'
max_length = 150
batch_size = 32
epochs = 10
lr = 1e-5
weight_decay = 1e-4

train_data = df[(df['type']=='train') & (df['fold']!=1)]
val_data = df[(df['type']=='train') & (df['fold']==1)]
test_data = df[df['type']=='test']

X_train = train_data[sentence]
y_train = train_data['label']

X_val = val_data[sentence]
y_val = val_data['label']

X_test = test_data[sentence]
y_test = test_data['label']

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_loader = MovieDataLoader(X_train, y_train, tokenizer, max_length, batch_size, type_='train')
val_loader = MovieDataLoader(X_val, y_val, tokenizer, max_length, batch_size, type_='val')
test_loader = MovieDataLoader(X_test, y_test, tokenizer, max_length, batch_size, type_='test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = BERT()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * epochs
)


if mode == 'train':
  trainer(model, train_loader, val_loader, criterion, optimizer, device, lr_scheduler)

elif mode == 'inference':
  ckpt = torch.load(pretrain)
  model.load_state_dict(ckpt)

inference(model, test_loader, criterion, device)
