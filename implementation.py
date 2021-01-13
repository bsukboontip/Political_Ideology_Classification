#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers')


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import pandas as pd
import numpy as np
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import rc
import re
from collections import defaultdict


# In[4]:


device = 'cuda' if torch.cuda.is_available()==True else 'cpu'
device = torch.device(device)
print(f'We are using device name "{device}"')


# In[5]:


# df = pd.read_csv("articles1.csv", encoding="ISO-8859-1")
df_1 = pd.read_csv("/content/drive/My Drive/ECE570/articles1.csv", encoding="ISO-8859-1")
df_2 = pd.read_csv("/content/drive/My Drive/ECE570/articles2.csv", encoding="ISO-8859-1")
df_3 = pd.read_csv("/content/drive/My Drive/ECE570/articles3.csv", encoding="ISO-8859-1")
print("FINISHED")


# In[6]:


df_1 = df_1[["id", "title", "publication", "content", "leaning"]]
df_2 = df_2[["id", "title", "publication", "content", "leaning"]]
df_3 = df_3[["id", "title", "publication", "content", "leaning"]]
df = pd.concat([df_1, df_2, df_3])

df = df[["id", "title", "publication", "content", "leaning"]]

df


# In[7]:


df["id"] = pd.to_numeric(df["id"], errors="coerce")
df.dropna(how="any", inplace=True)
df.dtypes
df = df[["title", "content", "leaning"]]


# In[8]:


def to_leaning(leaning):
#     print(leaning)
    if leaning == "left":
        return 0
    if leaning == "center":
        return 1
    if leaning == "right":
        return 2

def text_preprocessing(s):
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    
    return s

def remove_publication(s):
  s = s.replace("- The New York Times", " ")
  s = s.replace("- Breitbart", " ")
  return s



df['title'] = df.title.apply(text_preprocessing)
df['title'] = df.title.apply(remove_publication)
df['content'] = df.content.apply(text_preprocessing)
df['leaning'] = df.leaning.apply(to_leaning)
class_names = ['left', 'center', 'right']
df


# In[9]:


df1 = df.loc[df['leaning'] == 1]
sample_size = df1.size
sample_size = 27000

df0 = df.loc[df['leaning'] == 0]
df1 = df.loc[df['leaning'] == 1]
df2 = df.loc[df['leaning'] == 2]

df0 = df0.sample(int(sample_size))
df1 = df1.sample(int(sample_size))
df2 = df2.sample(int(sample_size))

dataframes = [df0, df1, df2]

df = pd.concat(dataframes)
df


# In[10]:


ax = sns.countplot(df.leaning)


# In[11]:


tokenizer = BertTokenizer.from_pretrained('bert-base-cased', truncation='longest_first')


# In[12]:


MAX_LEN_TITLE = 40
MAX_LEN_CONTENT = 600

class GPReviewDataset(Dataset):
    
  def __init__(self, titles, targets, contents, tokenizer, max_len_title, max_len_content):
    self.titles = titles
    self.targets = targets
    self.contents = contents
    self.tokenizer = tokenizer
    self.max_len_title = max_len_title
    self.max_len_content = max_len_content
    
  def __len__(self):
    return len(self.titles)

  def __getitem__(self, item):
    title = str(self.titles[item])
    content = str(self.contents[item])
    target = self.targets[item]
    
    encoding = self.tokenizer.encode_plus(
      title,
      add_special_tokens=True,
      max_length=self.max_len_title,
      return_token_type_ids=False,
#       pad_to_max_length=True,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
      truncation='longest_first',
    )
    
    return {
      'title_text': title,
      'content_text': content,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


# In[13]:


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df_train, df_test = train_test_split(
  df,
  test_size=0.1,
  random_state=RANDOM_SEED
)
df_val, df_test = train_test_split(
  df_test,
  test_size=0.5,
  random_state=RANDOM_SEED
)

df_train.shape, df_val.shape, df_test.shape


# In[14]:


def create_data_loader(df, tokenizer, max_len_title, max_len_content, batch_size):
  ds = GPReviewDataset(
    titles=df.title.to_numpy(),
    contents=df.content.to_numpy(),
    targets=df.leaning.to_numpy(),
    tokenizer=tokenizer,
    max_len_title=max_len_title,
    max_len_content=max_len_content
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

BATCH_SIZE = 16
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN_TITLE, MAX_LEN_CONTENT, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN_TITLE, MAX_LEN_CONTENT, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN_TITLE, MAX_LEN_CONTENT, BATCH_SIZE)


# In[15]:


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SentimentClassifier(nn.Module):
    
  def __init__(self, n_classes):
    
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased')

    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 24, 3)
    self.conv3 = nn.Conv2d(24, 32, 3)
    self.pool = nn.MaxPool2d(2, 2)

    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(64, n_classes)
    
  def forward(self, input_ids, attention_mask):
    
    # pooled_output is [4, 768]
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    # print(pooled_output.size)

    # [4, 1, 768]
    x = pooled_output.unsqueeze(1)
    shape = x.shape
    x = x.view(shape[0], 1, 32, int(x.shape[2]/32))

    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    # print(x.shape)
    # x = x.view(-1, 32)

    x = x.view(-1, 64)

    output = self.drop(x)
    
    return self.out(output)

model = SentimentClassifier(len(class_names))
model = model.to(device)
# input_ids = data['input_ids'].to(device)
# attention_mask = data['attention_mask'].to(device)

# print(input_ids.shape) # batch size x seq length
# print(attention_mask.shape) # batch size x seq length


# In[16]:


EPOCHS = 5

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[17]:


def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):

  model = model.train()
  losses = []
  correct_predictions = 0

  for d in data_loader:
        
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)
    
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
  return correct_predictions.double() / n_examples, np.mean(losses)


# In[18]:


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0

  with torch.no_grad():
        
    for d in data_loader:
        
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
    
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
        
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
        
  return correct_predictions.double() / n_examples, np.mean(losses)


# In[19]:


history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    len(df_train)
  )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(df_val)
  )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
    
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc


# In[20]:


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


# In[21]:


test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)
test_acc.item()


# In[22]:


def get_predictions(model, data_loader):
  model = model.eval()
  title_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
        
    for d in data_loader:
        
      texts = d["title_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
        
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
    
      _, preds = torch.max(outputs, dim=1)
      title_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
        
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()

  return title_texts, predictions, prediction_probs, real_values


# In[23]:


y_title_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

print(classification_report(y_test, y_pred, target_names=class_names))


# In[24]:


def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

