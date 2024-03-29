import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel, DistilBertModel

# Parameter - can be modified
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate =  4e-5
model_name = 'distilbert'

# Location of the model
ckpt_path="../assets/"
ckpt_name=ckpt_path+"saved_model.pt"

# Load the model and data
bert_model = BertModel.from_pretrained('monologg/kobert')
distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
bertmodel, vocab = get_pytorch_kobert_model()

data_path="../assets/" #your own path
data_name=data_path+"sentiment_dialogues.csv"
processed_data = pd.read_csv(data_name)

# Rename the column
processed_data.loc[(processed_data['감정'] == "불안"), '감정'] = 0  #불안 => 0
processed_data.loc[(processed_data['감정'] == "당황"), '감정'] = 1  #당황 => 1
processed_data.loc[(processed_data['감정'] == "분노"), '감정'] = 2  #분노 => 2
processed_data.loc[(processed_data['감정'] == "슬픔"), '감정'] = 3  #슬픔 => 3
processed_data.loc[(processed_data['감정'] == "중립"), '감정'] = 4  #중립 => 4
processed_data.loc[(processed_data['감정'] == "행복"), '감정'] = 5  #행복 => 5
processed_data.loc[(processed_data['감정'] == "혐오"), '감정'] = 6  #혐오 => 6

data_list = []
for q, label in zip(processed_data['발화'], processed_data['감정'])  :
    data = []
    data.append(q)
    data.append(str(label))
    data_list.append(data)

# Split the data
dataset_train, dataset_test = train_test_split(data_list, test_size=0.25)
dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.1)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
        
    def __len__(self):  
        return (len(self.labels))

# tokenizer
tokenizer = get_tokenizer()

tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
data_val = BERTDataset(dataset_val, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, num_workers=0)


# Create KoBERT model
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   # Modify (class number)
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.max_len = max_len
        self.batch_size = batch_size
        self.model_name = model_name
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        pooler = self.bert(input_ids = token_ids, attention_mask = attention_mask.float().to(token_ids.device))[0][:,0]
        # Distilbert does not return the pooler, so extract pooler from last_hidden_state
        # If this is BERT model, use this:
        # pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))[1]

        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Use CUDA if available    
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# Load the DistilBERT model
model = BERTClassifier(distilbert_model,  dr_rate=0.5).to(device)

# optimizer / scheduler
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# Calculate accuracy of the model
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
   
# Create Model
best_acc=0.0
best_loss=99999999

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    val_acc = 0.0
    model.train()

    for batch_id, (token_ids, valid_length,segment_ids,label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        # DistillBERT don't use token_type_ids(segment_ids)
        # If it uses BERT model, use this:
        # segment_ids = segment_ids.long().to(device) ## 
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    model.eval()
    for batch_id, (token_ids, valid_length,segment_ids,label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        # DistillBERT don't use token_type_ids(segment_ids)
        # If it uses BERT model, use this:
        # segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        with torch.no_grad():
          out = model(token_ids, valid_length)
        test_loss=loss_fn(out,label)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {} test loss {}".format(e+1, test_acc / (batch_id+1),test_loss.data.cpu().numpy()))

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(val_dataloader):
        token_ids = token_ids.long().to(device)
        # DistillBERT don't use token_type_ids(segment_ids)
        # If it uses BERT model, use this:
        # segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        with torch.no_grad():
          out = model(token_ids, valid_length)
        val_loss=loss_fn(out,label)
        val_acc += calc_accuracy(out, label)
    avg_val_acc = val_acc / len(val_dataloader)
    print("epoch {} val acc {}".format(e+1, avg_val_acc))

    if test_acc>best_acc and test_loss.data.cpu().numpy()<best_loss:
      torch.save({'epoch':e+1,
                  'model_state_dict':model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'loss':test_loss.data.cpu().numpy(),
                  'max_len': model.max_len,
                  'batch_size': model.batch_size,
                  'model_name':model.model_name},
                  ckpt_name)
      best_loss=test_loss.data.cpu().numpy()
      bset_acc=test_acc
      
      print('current best model saved. validation acc is : ', val_acc)