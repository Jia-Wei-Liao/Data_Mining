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

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup



class MovieDataset(Dataset):
  def __init__(self, sentences, labels, tokenizer, max_len):
    self.sentences = sentences
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.sentences)

  def __getitem__(self, i):
    sentence = str(self.sentences.iloc[i])
    label = self.labels.iloc[i]
    encoding = self.tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
    
    outputs = {
        'text': sentence,
        'index': encoding['input_ids'].flatten(),
        'mask': encoding['attention_mask'].flatten(),
        'label': torch.tensor(label, dtype=torch.long)
    }

    return outputs


def MovieDataLoader(X, y, tokenizer, max_len, batch_size, type_='train'):
  dataset = MovieDataset(
      sentences=X,
      labels=y,
      tokenizer=tokenizer,
      max_len=max_len
  )
  if type_ == 'train':
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

  else:
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

  return dataloader


class BERT(nn.Module):
  def __init__(self, n_classes=2):
    super(BERT, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')
    self.drop1 = nn.Dropout(p=0.4)
    self.dense1 = nn.Linear(self.bert.config.hidden_size, 128)
    self.relu1 = nn.ReLU()
    self.drop2 = nn.Dropout(p=0.4)
    self.dense2 = nn.Linear(128, n_classes)
  
  def forward(self, inputs, mask):
    bert_output = self.bert(
      input_ids=inputs,
      attention_mask=mask
    )
    x = self.drop1(bert_output['pooler_output'])
    x = self.dense1(x)
    x = self.relu1(x)
    x = self.drop2(x)
    outputs = self.dense2(x)

    return outputs


def train_step(model, train_loader, criterion, optimizer, device, lr_scheduler):
  n, total_loss, total_acc = 0, 0, 0
  model.train()

  for batch_data in tqdm.tqdm(train_loader):
    optimizer.zero_grad()

    inputs = batch_data['index'].to(device)
    labels = batch_data['label'].to(device)
    mask = batch_data['mask'].to(device)
    preds = model(inputs=inputs, mask=mask)
    loss = criterion(preds, labels)
    acc = torch.sum(torch.argmax(preds, dim=1) == labels)

    n += len(preds)
    total_loss += loss * len(preds)
    total_acc += acc

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()

  return {'loss': total_loss.item() / n, 'acc': total_acc.item() / n}


def val_step(model, val_loader, criterion, device):
  n, total_loss, total_acc = 0, 0, 0
  model.eval()

  with torch.no_grad():
    for batch_data in tqdm.tqdm(val_loader):
      inputs = batch_data['index'].to(device)
      labels = batch_data['label'].to(device)
      mask = batch_data['mask'].to(device)
      preds = model(inputs=inputs, mask=mask)
      loss = criterion(preds, labels)
      acc = torch.sum(torch.argmax(preds, dim=1) == labels)
      
      n += len(preds)
      total_loss += loss * len(preds)
      total_acc += acc

  return {'loss': total_loss.item() / n, 'acc': total_acc.item() / n}


def inference(model, test_loader, criterion, device):
  n, total_loss, total_acc = 0, 0, 0
  model.eval()

  with torch.no_grad():
    for batch_data in tqdm.tqdm(test_loader):
      inputs = batch_data['index'].to(device)
      labels = batch_data['label'].to(device)
      mask = batch_data['mask'].to(device)
      preds = model(inputs=inputs, mask=mask)
      loss = criterion(preds, labels)
      acc = torch.sum(torch.argmax(preds, dim=1) == labels)

      n += len(preds)
      total_loss += loss * len(preds)
      total_acc += acc

  print(f"loss: {total_loss.item() / n}, acc: {total_acc.item() / n}")


def trainer(model, train_loader, val_loader, criterion, optimizer, device, lr_scheduler):
  best_acc = 0
  record = pd.DataFrame(data=[], columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
  for ep in range(1, epochs+1):
    train_record = train_step(model, train_loader, criterion, optimizer, device, lr_scheduler)
    val_record = val_step(model, val_loader, criterion, device)
    print(f"train loss: {train_record['loss']}, train acc: {train_record['acc']}")
    print(f"val loss: {val_record['loss']}, val acc: {val_record['acc']}")
    
    record = record.append({
      'epoch': ep,
      'train_loss': train_record['loss'],
      'train_acc': train_record['acc'],
      'val_loss': val_record['loss'],
      'val_acc': val_record['acc']
    },
    ignore_index=True)
    os.makedirs('309652008/record', exist_ok=True)
    record.to_csv(f'309652008/record/record_{sentence}.csv', index=False)

    if val_record['acc'] > best_acc:
      save_root = '309652008/weight'
      os.makedirs(save_root, exist_ok=True)
      torch.save(model.state_dict(), os.path.join(save_root, f'best_model_{sentence}.bin'))
      best_acc = val_record['acc']

  return
